"""Helpers for the v2 ``/system-log`` API (UniFi Network 9.x+).

Response parsing, coverage reporting, and HTTP-status classification shared
by the event and stats managers. The controller answers ``/system-log/all``
with an envelope such as::

    [{"data": [...], "page_number": 0, "total_element_count": 42,
      "total_page_count": 1, "or_more": false}]

Coverage rule, used by every tool that reports it:

- the reader stopped while the controller still had events (a page or event
  cap, or a caller limit inside a page): truncated
- no total from the controller: completeness is unknown (``None``)
- collected < total: truncated
- collected >= total: complete, unless the controller flags the total as a
  lower bound (``or_more``), in which case completeness is unknown

Only events actually collected count; reaching a final page does not prove
the earlier pages were read.

Counts that are not non-negative integers are treated as absent, and a page
whose event list holds anything other than objects is unreadable, never
silently shortened. An empty page while the controller's counts say events
remain unread is contradictory and raises rather than ending the read.

The parser takes what ``ConnectionManager.request`` returns for a v2 request:
aiounifi's ``ApiRequestV2.decode`` wraps a JSON object body as a one-element
list, and the connection manager returns that list. A one-element list
holding a ``data``/``logs`` envelope is the envelope; any other list must be
a list of event rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Severities the system log accepts, lowest first.
EVENT_SEVERITIES: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")

#: Largest page the system log serves per request.
SYSTEM_LOG_PAGE_SIZE = 100

#: HTTP statuses that mean "this endpoint does not exist on this controller".
_ABSENT_STATUSES = frozenset({404, 405})

#: Fields of which a bare-list row needs at least one to count as an event:
#: the identity, key and time fields ``event_log_from_controller`` reads.
EVENT_ROW_FIELDS = frozenset({"_id", "id", "key", "event", "event_type", "time", "timestamp", "ts"})

#: aiounifi's request-error text: ``Call <url> received <status> <reason>``.
_AIOUNIFI_STATUS_RE = re.compile(r"^Call https?://\S+ received (\d{3})\b(?::|\s+([A-Za-z][A-Za-z ]*[A-Za-z]))?")

_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_BRACKETED_IPV6_RE = re.compile(r"\[[0-9A-Fa-f:.%]+\](?::\d{1,5})?")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")
# IPv6 in full eight-group form or compressed with "::". MAC addresses (six
# groups) and clock times (three groups) have neither, so they are kept.
_IPV6_RE = re.compile(
    r"(?<![0-9A-Za-z:])"
    r"(?:(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}"
    r"|(?:[0-9A-Fa-f]{1,4}:)*[0-9A-Fa-f]{0,4}::(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*)?)"
    r"(?![0-9A-Za-z:])"
)
# A host with a port: a dotted hostname (optionally fully qualified with a
# trailing dot) or localhost. "keys:123" and "BOGUS:1234" are not hosts.
_HOST_PORT_RE = re.compile(
    r"\b(?:localhost|(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.?)"
    r":\d{1,5}\b",
    re.IGNORECASE,
)
# The target named in aiohttp's connection error: "Cannot connect to host <target> ...".
_CONNECT_HOST_RE = re.compile(r"(\bconnect to host )(?!<address>)[^\s,;()]+", re.IGNORECASE)


class SystemLogResponseError(ValueError):
    """The system log answered with a shape that cannot be read as events."""


class SystemLogIncompleteError(RuntimeError):
    """A later system-log page failed after earlier pages were read.

    Distinct from a failure of the first request: it shows the endpoint
    exists, so it must never be read as "this controller has no system log".
    """


@dataclass
class SystemLogPage:
    """One parsed page of a system-log response."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    total: Optional[int] = None
    total_pages: Optional[int] = None
    #: ``or_more`` as sent; None when the page did not say.
    total_is_lower_bound: Optional[bool] = None


def _count_or_none(value: Any) -> Optional[int]:
    """A controller count, or None when it is missing, non-integer or negative."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _event_rows(rows: List[Any], *, require_event_fields: bool) -> List[Dict[str, Any]]:
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemLogResponseError(
                f"system log event list holds a {type(row).__name__} at position {index}, not an event object"
            )
        if require_event_fields and not EVENT_ROW_FIELDS.intersection(row):
            raise SystemLogResponseError(
                f"system log answered with an object at position {index} that is neither an envelope "
                "nor an event (no id, key or time field)"
            )
    return list(rows)


def _is_envelope(value: Any) -> bool:
    return isinstance(value, dict) and ("data" in value or "logs" in value)


def parse_system_log_page(response: Any) -> SystemLogPage:
    """Parse a ``/system-log`` response into events plus the controller's counts.

    Accepts a one-element list holding a ``data``/``logs`` envelope (a JSON
    object body as delivered by the connection manager), the envelope itself,
    or a list of event rows (total unknown). Raises ``SystemLogResponseError``
    for anything else, so a malformed answer is never mistaken for an empty one
    or for an event.
    """
    envelope: Any = response
    if isinstance(response, list) and len(response) == 1 and _is_envelope(response[0]):
        envelope = response[0]

    if isinstance(envelope, dict):
        if not _is_envelope(envelope):
            raise SystemLogResponseError(f"system log response has no event list (keys: {sorted(envelope)[:10]})")
        data = envelope.get("data", envelope.get("logs"))
        if not isinstance(data, list):
            raise SystemLogResponseError("system log response's event list is not a list")
        or_more = envelope.get("or_more", envelope.get("orMore"))
        return SystemLogPage(
            events=_event_rows(data, require_event_fields=False),
            total=_count_or_none(envelope.get("total_element_count", envelope.get("totalElementCount"))),
            total_pages=_count_or_none(envelope.get("total_page_count", envelope.get("totalPageCount"))),
            total_is_lower_bound=or_more if isinstance(or_more, bool) else None,
        )
    if isinstance(envelope, list):
        return SystemLogPage(events=_event_rows(envelope, require_event_fields=True))
    raise SystemLogResponseError(f"system log response is a {type(response).__name__}, not an event list")


class PageTracker:
    """Carry the controller's counts across pages and decide whether to read on.

    Progress is the number of events actually collected, plus ``base`` for a
    read that starts at a later page (``first page number * page size``, the
    one place the server's page size has to be assumed). Counts and
    ``or_more`` are carried forward from earlier pages when a later page omits
    them.
    """

    def __init__(self, page_size: int, base: int = 0) -> None:
        self.page_size = page_size
        self.base = base
        self.collected = 0
        self.total: Optional[int] = None
        self.total_pages: Optional[int] = None
        self.total_is_lower_bound = False

    def observe(self, page: SystemLogPage) -> None:
        self.collected += len(page.events)
        if page.total is not None:
            self.total = page.total
        if page.total_pages is not None:
            self.total_pages = page.total_pages
        if page.total_is_lower_bound is not None:
            self.total_is_lower_bound = page.total_is_lower_bound

    @property
    def served(self) -> int:
        """Events the controller has served up to the end of the last observed page."""
        return self.base + self.collected

    def has_next_page(self, page: SystemLogPage, page_number: int) -> bool:
        """True when the controller may hold events after ``page`` (0-based ``page_number``).

        Call after ``observe(page)``. A short page ends the read only when no
        count says more exists, so a server that serves smaller pages than
        requested is still read through. A reached exact total ends the read
        even if the page count claims more pages.

        Raises:
            SystemLogResponseError: an empty page while the total or page count
                says events remain unread.
        """
        more_by_total = self.total is not None and self.served < self.total
        more_by_pages = self.total_pages is not None and page_number + 1 < self.total_pages
        total_reached = self.total is not None and not more_by_total and not self.total_is_lower_bound
        if not page.events:
            # A lower-bound total never confirms the end: only a reached exact
            # total overrides a page count that says more pages follow.
            if more_by_total or (more_by_pages and not total_reached):
                raise SystemLogResponseError(
                    f"system log page {page_number + 1} is empty, but its counts say events remain "
                    f"(read {self.served}, total {self.total}, pages {self.total_pages})"
                )
            return False
        if more_by_total:
            return True
        if total_reached:
            return False
        if more_by_pages:
            return True
        if self.total_pages is not None and not self.total_is_lower_bound:
            return False
        return len(page.events) >= self.page_size


def coverage(
    collected: int,
    total: Optional[int],
    total_is_lower_bound: bool = False,
    stopped_early: bool = False,
) -> Dict[str, Optional[bool]]:
    """Apply the coverage rule in the module docstring."""
    if stopped_early:
        return {"complete": False, "truncated": True}
    if total is None:
        return {"complete": None, "truncated": None}
    if collected < total:
        return {"complete": False, "truncated": True}
    if total_is_lower_bound:
        return {"complete": None, "truncated": None}
    return {"complete": True, "truncated": False}


def _http_code(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= 599:
        return None
    return value


def http_status(error: BaseException) -> Optional[int]:
    """HTTP status for a controller request error, or None when it does not carry one.

    Structured values win: a ``status`` attribute, then ``errorCode``/``statusCode``
    in an error-body dict (how aiounifi reports v2 errors). Text counts only in
    aiounifi's request-error form anchored at the start,
    ``Call <url> received <status> ...``; numbers anywhere else are ignored.
    """
    status = _http_code(getattr(error, "status", None))
    if status is not None:
        return status
    for arg in getattr(error, "args", ()):
        if isinstance(arg, dict):
            code = _http_code(arg.get("errorCode", arg.get("statusCode")))
            if code is not None:
                return code
    match = _AIOUNIFI_STATUS_RE.match(str(error))
    return _http_code(int(match.group(1))) if match else None


def is_endpoint_absent(error: BaseException) -> bool:
    """True when *error* says the endpoint does not exist (404/405)."""
    return http_status(error) in _ABSENT_STATUSES


def scrub_addresses(text: str) -> str:
    """Replace URLs, IP addresses, hosts with ports and connection targets in *text*.

    Leaves identifiers that only look address-like alone: ``keys:123``, MAC
    addresses and clock times.
    """
    text = _URL_RE.sub("<url>", text)
    text = _BRACKETED_IPV6_RE.sub("<address>", text)
    text = _IPV4_RE.sub("<address>", text)
    text = _IPV6_RE.sub("<address>", text)
    text = _HOST_PORT_RE.sub("<address>", text)
    return _CONNECT_HOST_RE.sub(r"\1<address>", text)


def controller_message(error: BaseException) -> str:
    """The controller's own error message when the exception carries one, address-scrubbed.

    Falls back to ``describe_request_error``.
    """
    for arg in getattr(error, "args", ()):
        if isinstance(arg, dict) and isinstance(arg.get("message"), str):
            return scrub_addresses(arg["message"])
    return describe_request_error(error)


def describe_request_error(error: BaseException) -> str:
    """Short caller-facing description of a request error, with addresses scrubbed.

    This is the boundary for controller error text: every branch passes
    through ``scrub_addresses``.
    """
    status = http_status(error)
    for arg in getattr(error, "args", ()):
        if isinstance(arg, dict) and isinstance(arg.get("message"), str):
            message = scrub_addresses(arg["message"])
            return f"HTTP {status}: {message}" if status else message
    text = str(error)
    match = _AIOUNIFI_STATUS_RE.match(text)
    if match and status is not None:
        reason = match.group(2) or ""
        return scrub_addresses(f"HTTP {status} {reason}".strip())
    text = scrub_addresses(text).strip()
    return text or type(error).__name__


def safe_error_text(error: BaseException) -> str:
    """Error text for a tool result, with addresses scrubbed whatever the exception type.

    ``ValueError`` keeps its own wording (this server's validation messages)
    but is scrubbed too: a pydantic ``ValidationError`` is a ``ValueError`` and
    echoes the controller-supplied input it rejected. Anything else is
    described through ``describe_request_error``.
    """
    if isinstance(error, ValueError):
        return scrub_addresses(str(error)).strip() or type(error).__name__
    return describe_request_error(error)
