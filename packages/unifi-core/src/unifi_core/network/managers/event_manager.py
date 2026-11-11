"""Event Manager for UniFi Network MCP server.

Manages event log and alarm operations using the v2 system-log API
(UniFi Network 10.x+). Falls back to legacy /stat/event for older controllers.
"""

import logging
import re
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from aiounifi.models.api import ApiRequest, ApiRequestV2
from aiounifi.models.message import MessageKey

from unifi_core.mac import mac_equal
from unifi_core.network.managers.connection_manager import ConnectionManager
from unifi_core.network.system_log import (
    EVENT_SEVERITIES,
    SYSTEM_LOG_PAGE_SIZE,
    PageTracker,
    SystemLogIncompleteError,
    SystemLogPage,
    SystemLogResponseError,
    controller_message,
    coverage,
    describe_request_error,
    http_status,
    parse_system_log_page,
)

logger = logging.getLogger("unifi-network-mcp")


# ---------------------------------------------------------------------------
# EventBuffer
# ---------------------------------------------------------------------------


class EventBuffer:
    """Ring buffer for recent network events received via websocket.

    Same contract as protect/access EventBuffer. Events are stored as plain
    dicts with a ``_buffered_at`` timestamp for TTL-based lazy expiration.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._ttl = ttl_seconds

    def add(self, event: dict[str, Any]) -> None:
        """Add *event* to the buffer, stamping it with the current time."""
        event = {**event, "_buffered_at": time.time()}
        self._buffer.append(event)

    def get_recent(
        self,
        event_type: str | None = None,
        mac: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return most-recent-first events, filtered and TTL-aged."""
        cutoff = time.time() - self._ttl
        out: list[dict[str, Any]] = []
        for event in reversed(self._buffer):
            if event.get("_buffered_at", 0) < cutoff:
                continue
            if event_type and event.get("key") != event_type:
                continue
            if mac and not mac_equal(event.get("mac"), mac):
                continue
            out.append(event)
            if limit and len(out) >= limit:
                break
        return out

    def clear(self) -> None:
        """Remove all events from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


#: System-log categories this server knows, with descriptions. They are the
#: default query when a caller names none. The controller may accept categories
#: missing from this list, so caller-supplied categories are checked for shape
#: only and the controller's verdict is passed back.
KNOWN_EVENT_CATEGORIES: Dict[str, str] = {
    "CLIENT_DEVICES": "Client device connect/disconnect events",
    "INTERNET_AND_WAN": "Internet outage, failover, and performance",
    "POWER": "PoE, power supply, and UPS events",
    "SECURITY": "Firewall, IPS, honeypot events",
    "UNIFI_DEVICES": "Device adoption, discovery, reconnection",
    "SOFTWARE_UPDATES": "Firmware update events",
    "UNIFI_ETHERNET_PORTS": "Port events (STP, storms, errors)",
    "VPN": "VPN client and site-to-site events",
}

# Default categories for the system-log v2 API
_DEFAULT_CATEGORIES = list(KNOWN_EVENT_CATEGORIES)

# Default severities
_DEFAULT_SEVERITIES = list(EVENT_SEVERITIES)

#: Bound on system-log pages read for one list_events call. Beyond it the result
#: is reported as truncated rather than fetched without limit.
EVENT_LIST_MAX_PAGES = 50

_CATEGORY_SHAPE = re.compile(r"[A-Z][A-Z0-9_]*")

_LEGACY_PREFIX_ERROR = (
    "event_type '{value}' is a legacy EVT_* key or prefix, which the system log on current UniFi "
    "Network controllers does not accept. Pass an exact event key (unifi_get_event_types lists "
    "recently observed keys) or filter by categories instead."
)


def _validate_categories(categories: Optional[List[str]]) -> Optional[List[str]]:
    """Check caller-supplied categories for shape; the controller decides membership."""
    if categories is None:
        return None
    if not isinstance(categories, list):
        raise ValueError("categories must be a list of category names")
    known = ", ".join(_DEFAULT_CATEGORIES)
    validated: List[str] = []
    for category in categories:
        if not isinstance(category, str) or not _CATEGORY_SHAPE.fullmatch(category) or category.startswith("EVT_"):
            raise ValueError(
                f"Invalid category {category!r}: categories are upper-case identifiers such as SECURITY, "
                f"not event keys. Known categories: {known}."
            )
        if category not in validated:
            validated.append(category)
    return validated or None


def _validate_query_window(within: Any, limit: Any, start: Any, event_type: Any) -> None:
    """Reject arguments that would produce an empty or inverted query window."""

    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    if not is_int(within) or within < 1:
        raise ValueError(f"within_hours must be an integer of at least 1, got {within!r}")
    if not is_int(limit) or limit < 1:
        raise ValueError(f"limit must be an integer of at least 1, got {limit!r}")
    if not is_int(start) or start < 0:
        raise ValueError(f"start must be an integer of at least 0, got {start!r}")
    if event_type is not None and not isinstance(event_type, str):
        raise ValueError(f"event_type must be a string, got {type(event_type).__name__}")


def _page_is_misaligned(page: SystemLogPage, tracker: PageTracker) -> bool:
    """True when a page reached by jumping to ``start // page_size`` cannot be where the jump assumed.

    Judged from the controller's counts against the events the page should
    hold: with an exact total, exactly ``min(page_size, total - offset)``; with
    a lower-bound total, at least that many; with only a page count, a full page
    unless it is the last. With no counts there is nothing to check against.
    A short page that the counts accept still needs the page-0 stride check
    in ``_get_events_v2``: counts cannot tell a short final page from a
    server that pages by fewer events than requested.
    """
    offset = tracker.served - len(page.events)
    if tracker.total is not None:
        expected = min(tracker.page_size, max(tracker.total - offset, 0))
        if tracker.total_is_lower_bound:
            return len(page.events) < expected
        return len(page.events) != expected
    if tracker.total_pages is not None:
        page_number = offset // tracker.page_size
        return page_number + 1 < tracker.total_pages and len(page.events) < tracker.page_size
    return False


def _validate_severities(severities: Optional[List[str]]) -> Optional[List[str]]:
    """Check caller-supplied severities against the system log's fixed scale."""
    if severities is None:
        return None
    if not isinstance(severities, list):
        raise ValueError("severities must be a list of severity names")
    validated: List[str] = []
    for severity in severities:
        if severity not in EVENT_SEVERITIES:
            raise ValueError(f"Invalid severity {severity!r}. Valid severities: {', '.join(EVENT_SEVERITIES)}.")
        if severity not in validated:
            validated.append(severity)
    return validated or None


class EventManager:
    """Manages event log operations on the UniFi Controller.

    REST query surface (existing) plus websocket buffer + per-subscriber
    fan-out (new in Phase 4B). Mirrors the protect/access EventManager
    contract: ``start_listening`` opens the websocket and registers an
    ``_on_ws_event`` handler with ``aiounifi.Controller.messages`` (the
    upstream MessageHandler subscribe pattern), events are normalized into
    a stable network-event dict shape, written to a TTL ring buffer, and
    fanned out to every callback registered via ``add_subscriber``.
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        config: dict | None = None,
    ) -> None:
        cfg = config or {}
        self._connection = connection_manager
        self._cm = connection_manager  # alias mirroring protect/access naming
        self._use_v2: bool | None = None  # Auto-detect on first call
        # Why the v2 probe failed, when it did. Newer controllers have removed the
        # legacy paths this falls back to, so a failed probe resurfaces later as a
        # bare 404 from an endpoint the caller never asked for.
        self._v2_probe_error: str | None = None
        self._buffer = EventBuffer(
            max_size=int(cfg.get("buffer_size", 100)),
            ttl_seconds=int(cfg.get("buffer_ttl_seconds", 300)),
        )
        self._subscribers: list[Callable[[dict], None]] = []
        self._ws_unsub: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Websocket lifecycle
    # ------------------------------------------------------------------

    async def start_listening(self) -> None:
        """Open the aiounifi websocket and subscribe to event/alert messages.

        aiounifi exposes events through ``Controller.messages`` (a
        ``MessageHandler``). ``messages.subscribe(callback, message_filter)``
        returns an unsubscribe callable. We filter to ``EVENT`` + ``ALERT``
        so only event-log payloads flow through.
        """
        controller = self._cm.controller
        await controller.start_websocket()
        self._ws_unsub = controller.messages.subscribe(
            self._on_ws_event,
            (MessageKey.EVENT, MessageKey.ALERT),
        )
        logger.info("[network-event-mgr] websocket subscription started")

    async def stop_listening(self) -> None:
        """Unsubscribe from the websocket message handler."""
        if self._ws_unsub is not None:
            try:
                self._ws_unsub()
            except Exception:
                logger.debug("[network-event-mgr] error unsubscribing", exc_info=True)
            self._ws_unsub = None
            logger.info("[network-event-mgr] websocket subscription stopped")

    # ------------------------------------------------------------------
    # Event ingestion + fan-out
    # ------------------------------------------------------------------

    def _on_ws_event(self, event_obj: Any) -> None:
        """Normalize, buffer, and fan out a websocket event.

        Accepts either an aiounifi ``Message`` (with ``.data``), a raw dict,
        or anything exposing a ``.raw`` attribute. Subscribers receive the
        buffered dict (with ``_buffered_at`` stamped).
        """
        try:
            event = self._normalize_event(event_obj)
            if event is None:
                return
            self._buffer.add(event)
            stored = next(iter(self._buffer.get_recent(limit=1)), event)
            for cb in list(self._subscribers):
                try:
                    cb(stored)
                except Exception:
                    logger.debug(
                        "[network-event-mgr] subscriber callback failed",
                        exc_info=True,
                    )
        except Exception:
            logger.debug("[network-event-mgr] error processing ws event", exc_info=True)

    def _normalize_event(self, event_obj: Any) -> dict | None:
        """Convert an aiounifi Message / dict / Event into the stable shape."""
        if isinstance(event_obj, dict):
            raw = event_obj
        else:
            # aiounifi.Message exposes the event payload via `.data`
            raw = getattr(event_obj, "data", None)
            if raw is None:
                raw = getattr(event_obj, "raw", None)
        if not isinstance(raw, dict):
            return None
        return {
            "id": raw.get("_id") or raw.get("id"),
            "key": raw.get("key"),
            "msg": raw.get("msg"),
            "severity": raw.get("severity"),
            "time": raw.get("time"),
            "mac": raw.get("user") or raw.get("ap") or raw.get("sw") or raw.get("gw"),
            "ip": raw.get("ip"),
        }

    # ------------------------------------------------------------------
    # Subscriber management
    # ------------------------------------------------------------------

    def add_subscriber(self, cb: Callable[[dict], None]) -> Callable[[], None]:
        """Register *cb* to receive every event after it is buffered.

        Returns an unsubscribe callable.
        """
        self._subscribers.append(cb)

        def _unsub() -> None:
            try:
                self._subscribers.remove(cb)
            except ValueError:
                pass

        return _unsub

    # ------------------------------------------------------------------
    # Buffer access
    # ------------------------------------------------------------------

    def get_recent_from_buffer(
        self,
        event_type: str | None = None,
        mac: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        return self._buffer.get_recent(event_type=event_type, mac=mac, limit=limit)

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    async def _detect_api_version(self) -> bool:
        """Detect whether the controller supports the v2 system-log API.

        Returns True for v2, False for legacy.
        """
        try:
            now_ms = int(time.time() * 1000)
            one_hour_ago_ms = now_ms - (3600 * 1000)

            api_request = ApiRequestV2(
                method="post",
                path="/system-log/count",
                data={
                    "timestampFrom": one_hour_ago_ms,
                    "timestampTo": now_ms,
                    "severities": _DEFAULT_SEVERITIES,
                    "categories": _DEFAULT_CATEGORIES,
                    "type": "GENERAL",
                },
            )
            await self._connection.request(api_request)
            logger.info("[events] Using v2 system-log API")
            self._v2_probe_error = None
            return True
        except Exception as probe_error:
            # Log why, not just that. A probe that fails for a reason other than
            # "this controller predates v2" sends every later call down a legacy
            # path that modern controllers answer with 404, and without this the
            # only visible symptom is that 404.
            self._v2_probe_error = f"{type(probe_error).__name__}: {describe_request_error(probe_error)}"
            logger.warning(
                "[events] v2 system-log probe failed, falling back to legacy /stat/event API: %s",
                probe_error,
            )
            return False

    def _explain_legacy_failure(self, endpoint: str, error: Exception) -> Exception:
        """Attach the v2 probe failure to a legacy error, when it caused the fallback.

        Returns the original error untouched on a controller that legitimately has
        no v2 API — there is nothing extra to say about that case.
        """
        if not self._v2_probe_error:
            return error
        return RuntimeError(
            f"{endpoint} failed ({describe_request_error(error)}). This controller was put on the legacy events API "
            f"because the v2 system-log probe failed: {self._v2_probe_error}. "
            "On current UniFi Network versions the legacy endpoint no longer exists, so the "
            "underlying problem is the failed v2 probe, not the missing legacy path."
        )

    async def _ensure_api_version(self) -> None:
        """Detect API version on first call."""
        if self._use_v2 is None:
            self._use_v2 = await self._detect_api_version()

    async def get_events(
        self,
        within: int = 24,
        limit: int = 100,
        start: int = 0,
        event_type: Optional[str] = None,
        categories: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get events from the controller.

        Uses the v2 system-log API on modern controllers (10.x+),
        falls back to legacy /stat/event for older versions.

        Args:
            within: Hours to look back (default 24).
            limit: Maximum number of events to return (default 100).
            start: Offset for pagination (default 0).
            event_type: Optional filter for specific event type.
            categories: Optional list of categories to filter (v2 only).
            severities: Optional list of severities to filter (v2 only).

        Returns:
            List of event objects.
        """
        page = await self.get_events_page(
            within=within,
            limit=limit,
            start=start,
            event_type=event_type,
            categories=categories,
            severities=severities,
        )
        return page["events"]

    async def get_events_page(
        self,
        within: int = 24,
        limit: int = 100,
        start: int = 0,
        event_type: Optional[str] = None,
        categories: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get events plus the controller's total and how much of it was returned.

        Arguments are validated before any request: ``within`` and ``limit`` at
        least 1, ``start`` at least 0, ``event_type`` a string, categories by
        shape and severities against the system log's scale. On the v2 system
        log a legacy ``EVT_*`` event_type is rejected rather than sent, since the
        controller answers it with HTTP 400. Categories and severities need the
        system log; a legacy-API controller rejects them.

        Returns:
            Dict with ``events``, ``source`` (``"system-log"`` or ``"legacy"``),
            ``total`` (controller's count for the whole filtered window, or None),
            ``complete``/``truncated`` (see ``unifi_core.network.system_log``;
            never complete when ``start`` > 0), ``has_more`` (True when events
            exist after this page, False only when an exact total is reached,
            None when the total is absent or a reached lower bound), and the
            ``categories`` and ``severities`` actually sent (None on the legacy API).
        """
        _validate_query_window(within, limit, start, event_type)
        categories = _validate_categories(categories)
        severities = _validate_severities(severities)

        await self._ensure_api_version()

        if self._use_v2:
            if event_type and event_type.upper().startswith("EVT_"):
                raise ValueError(_LEGACY_PREFIX_ERROR.format(value=event_type))
            return await self._get_events_v2(within, limit, start, event_type, categories, severities)

        if categories or severities:
            raise ValueError(
                "categories and severities require the system log; this controller is on the legacy "
                "events API (/stat/event), which cannot filter by them. Omit them, or filter by event_type."
            )

        events = await self._get_events_legacy(within, limit, start, event_type)
        return {
            "events": events,
            "source": "legacy",
            "total": None,
            "complete": None,
            "truncated": None,
            "has_more": None,
            "categories": None,
            "severities": None,
        }

    async def _get_events_v2(
        self,
        within: int,
        limit: int,
        start: int,
        event_type: Optional[str],
        categories: Optional[List[str]],
        severities: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Get events using the v2 system-log API.

        A read with ``start > 0`` jumps straight to page ``start // page_size``,
        which assumes the server numbers pages by the requested page size. The
        page it gets back is verified before it is used:

        - If its length does not fit the controller's counts, the jump was wrong.
        - If it is non-empty but shorter than the page size, the counts cannot
          tell a short final page from a server that pages by fewer events, so
          page 0 is fetched once: a short page 0 means the server's pages are
          shorter than requested; a full page 0 means the jump was sound.
        - A full or empty jumped page is used as it is, with no extra request.

        On a wrong jump the read restarts from page 0 (reusing the page 0
        already fetched), reads on with the server's page length and discards
        the first ``start`` events locally. A restart costs the pages from 0 up
        to the requested window; every request, the check included, counts
        toward ``EVENT_LIST_MAX_PAGES``, and a read that hits the cap is
        truncated. A failed check fails the read.
        """
        applied_categories = categories or _DEFAULT_CATEGORIES
        applied_severities = severities or _DEFAULT_SEVERITIES
        result: Dict[str, Any] = {
            "events": [],
            "source": "system-log",
            "total": None,
            "complete": None,
            "truncated": None,
            "has_more": None,
            "categories": list(applied_categories),
            "severities": list(applied_severities),
        }
        try:
            now_ms = int(time.time() * 1000)
            from_ms = now_ms - (within * 3600 * 1000)
            page_size = min(limit, SYSTEM_LOG_PAGE_SIZE)
            page_number = start // page_size
            page_offset = start % page_size

            payload: Dict[str, Any] = {
                "timestampFrom": from_ms,
                "timestampTo": now_ms,
                "severities": list(applied_severities),
                "categories": list(applied_categories),
                "type": "GENERAL",
                "pageNumber": page_number,
                "pageSize": page_size,
                "searchText": "",
            }

            if event_type:
                # ``keys`` is the controller's exact event-key filter. ``searchText``
                # searches rendered message text and silently returns the wrong result
                # for values such as CLIENT_DISCONNECTED_WIRELESS_2.
                payload["keys"] = [event_type]

            events: List[Dict[str, Any]] = []
            tracker = PageTracker(page_size, base=page_number * page_size)
            check_alignment = page_number > 0
            pages_read = 0
            more = True

            async def read_page(number: int) -> SystemLogPage:
                payload["pageNumber"] = number
                api_request = ApiRequestV2(method="post", path="/system-log/all", data=payload.copy())
                return parse_system_log_page(await self._connection.request(api_request))

            while more and len(events) < limit + page_offset and pages_read < EVENT_LIST_MAX_PAGES:
                requested = page_number
                try:
                    page = await read_page(page_number)
                    pages_read += 1
                    tracker.observe(page)
                    if check_alignment:
                        check_alignment = False
                        first = None
                        misaligned = _page_is_misaligned(page, tracker)
                        if misaligned or 0 < len(page.events) < page_size:
                            requested = 0
                            first = await read_page(0)
                            pages_read += 1
                            if page.events and not first.events:
                                raise SystemLogResponseError(
                                    f"system log page 1 is empty, but page {page_number + 1} held events"
                                )
                            # Page 0 decides: a full one means pages are numbered by the
                            # requested size, so the jumped page stands whatever the counts
                            # suggested; a short one means the read must start over.
                            misaligned = len(first.events) < page_size
                        if misaligned:
                            logger.info(
                                "[events] system-log page %s does not match page size %s; re-reading from page 0",
                                page_number,
                                page_size,
                            )
                            stride = len(first.events) if 0 < len(first.events) < page_size else page_size
                            tracker = PageTracker(stride)
                            tracker.observe(first)
                            events, page, page_number, page_offset = [], first, 0, start
                    more = tracker.has_next_page(page, page_number)
                except Exception as page_error:
                    if pages_read == 0:
                        raise self._explain_rejected_filter(page_error, categories, event_type) from page_error
                    raise SystemLogIncompleteError(
                        f"system-log page {requested + 1} failed after {len(events)} events were read "
                        f"({describe_request_error(page_error)})"
                    ) from page_error
                events.extend(page.events)
                page_number += 1

            fetched_past_limit = len(events) > page_offset + limit
            events = events[page_offset : page_offset + limit]
            total, lower_bound = tracker.total, tracker.total_is_lower_bound
            result["events"] = events
            result["total"] = total
            result.update(coverage(len(events), total, lower_bound, stopped_early=more or fetched_past_limit))
            if start > 0 and result["complete"]:
                # A later page never covers the whole window, even when it is the last one.
                result["complete"] = False
            served = start + len(events)
            if total is not None and served < total:
                result["has_more"] = True
            elif total is not None and not lower_bound:
                result["has_more"] = False
            elif fetched_past_limit:
                result["has_more"] = True
            return result
        except Exception as e:
            logger.error("Error getting events (v2): %s", e)
            raise

    @staticmethod
    def _explain_rejected_filter(
        error: Exception, categories: Optional[List[str]], event_type: Optional[str]
    ) -> Exception:
        """Turn a first-page failure into a caller-safe error.

        A controller 400 on caller-supplied filters becomes an actionable
        ``ValueError``; this server's own response errors pass through; any
        other failure is described without the controller's address.
        """
        if isinstance(error, SystemLogResponseError):
            return error
        if http_status(error) != 400 or not (categories or event_type):
            return RuntimeError(f"system-log request failed ({describe_request_error(error)})")
        hints = []
        if categories:
            hints.append(
                f"Categories known to this server: {', '.join(_DEFAULT_CATEGORIES)} "
                "(the controller may accept others, and may reject some of these)."
            )
        if event_type:
            hints.append("event_type must be an exact event key; unifi_get_event_types lists recently observed keys.")
        return ValueError(f"The controller rejected the event filter: {controller_message(error)}. {' '.join(hints)}")

    async def _get_events_legacy(
        self,
        within: int,
        limit: int,
        start: int,
        event_type: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Get events using the legacy /stat/event API."""
        try:
            payload: Dict[str, Any] = {
                "within": within,
                "_limit": min(limit, 3000),
                "_start": start,
            }
            if event_type:
                payload["type"] = event_type

            api_request = ApiRequest(method="post", path="/stat/event", data=payload)
            response = await self._connection.request(api_request)

            if isinstance(response, list):
                return response
            if isinstance(response, dict):
                return response.get("data", [])
            return []
        except Exception as e:
            logger.error("Error getting events (legacy): %s", e)
            raise self._explain_legacy_failure("/stat/event", e) from e

    async def get_alarms(
        self,
        archived: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get active alarms/alerts from the controller.

        Uses v2 system-log/critical on modern controllers,
        falls back to legacy /stat/alarm for older versions.
        """
        await self._ensure_api_version()

        if self._use_v2:
            return await self._get_alarms_v2(archived, limit)
        return await self._get_alarms_legacy(archived, limit)

    async def _get_alarms_v2(self, archived: bool, limit: int) -> List[Dict[str, Any]]:
        """Get alarms using the v2 system-log/critical API."""
        try:
            now_ms = int(time.time() * 1000)
            # Look back 30 days for alarms
            from_ms = now_ms - (30 * 24 * 3600 * 1000)

            payload: Dict[str, Any] = {
                "timestampFrom": from_ms,
                "timestampTo": now_ms,
                "severities": ["HIGH", "VERY_HIGH"],
                "categories": _DEFAULT_CATEGORIES,
                "type": "GENERAL",
                "pageNumber": 0,
                "pageSize": min(limit, 100),
                "searchText": "",
            }

            api_request = ApiRequestV2(
                method="post",
                path="/system-log/critical",
                data=payload,
            )
            response = await self._connection.request(api_request)

            # V2 response comes as [{"data": [...], "total_element_count": N}] or {"data": [...]}
            if isinstance(response, list) and response and isinstance(response[0], dict) and "data" in response[0]:
                return response[0]["data"][:limit]
            if isinstance(response, dict):
                return response.get("data", response.get("logs", []))[:limit]
            if isinstance(response, list):
                return response[:limit]
            return []
        except Exception as e:
            logger.error("Error getting alarms (v2): %s", e)
            raise

    async def _get_alarms_legacy(self, archived: bool, limit: int) -> List[Dict[str, Any]]:
        """Get alarms using the legacy /stat/alarm API."""
        try:
            path = "/stat/alarm"
            if archived:
                path = "/stat/alarm?archived=true"

            api_request = ApiRequest(method="get", path=path)
            response = await self._connection.request(api_request)

            alarms = (
                response
                if isinstance(response, list)
                else response.get("data", [])
                if isinstance(response, dict)
                else []
            )
            return alarms[:limit]
        except Exception as e:
            logger.error("Error getting alarms (legacy): %s", e)
            raise self._explain_legacy_failure("/stat/alarm", e) from e

    def get_event_type_prefixes(self) -> List[Dict[str, str]]:
        """Get legacy event type prefixes for backward compatibility."""
        return [
            {"prefix": "EVT_SW_", "description": "Switch events"},
            {"prefix": "EVT_AP_", "description": "Access Point events"},
            {"prefix": "EVT_GW_", "description": "Gateway events"},
            {"prefix": "EVT_LAN_", "description": "LAN events"},
            {"prefix": "EVT_WU_", "description": "WLAN User events (connect/disconnect)"},
            {"prefix": "EVT_WG_", "description": "WLAN Guest events"},
            {"prefix": "EVT_IPS_", "description": "IPS/IDS security events"},
            {"prefix": "EVT_AD_", "description": "Admin events"},
            {"prefix": "EVT_DPI_", "description": "Deep Packet Inspection events"},
        ]

    async def get_event_types(self) -> List[Dict[str, Any]]:
        """Get recently observed exact event keys for filtering.

        Modern controllers expose exact enum keys rather than the legacy
        ``EVT_*`` prefix catalog, so discover usable values from recent events.
        """
        events = await self.get_events(within=168, limit=1000)
        counts: Dict[str, int] = {}
        for event in events:
            key = event.get("key") or event.get("event")
            if isinstance(key, str) and key:
                counts[key] = counts.get(key, 0) + 1

        return [
            {
                "key": key,
                # Retain the old field for consumers of the existing response
                # shape; its value is now an exact key, not a prefix.
                "prefix": key,
                "description": "Exact event key observed in the 1,000 most recent events within the last 7 days",
                "observed_count": counts[key],
            }
            for key in sorted(counts)
        ]

    def get_event_categories(self) -> List[Dict[str, str]]:
        """Get available event categories for v2 API filtering."""
        return [
            {"category": category, "description": description}
            for category, description in KNOWN_EVENT_CATEGORIES.items()
        ]

    async def archive_alarm(self, alarm_id: str) -> bool:
        """Archive an alarm (mark as resolved)."""
        try:
            api_request = ApiRequest(
                method="post",
                path="/cmd/evtmgr",
                data={"cmd": "archive-alarm", "_id": alarm_id},
            )
            await self._connection.request(api_request)
            logger.info("Archived alarm %s", alarm_id)
            return True
        except Exception as e:
            logger.error("Error archiving alarm %s: %s", alarm_id, e)
            raise

    async def archive_all_alarms(self) -> bool:
        """Archive all active alarms."""
        try:
            api_request = ApiRequest(
                method="post",
                path="/cmd/evtmgr",
                data={"cmd": "archive-all-alarms"},
            )
            await self._connection.request(api_request)
            logger.info("Archived all alarms")
            return True
        except Exception as e:
            logger.error("Error archiving all alarms: %s", e)
            raise
