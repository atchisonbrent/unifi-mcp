import copy
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from aiounifi.models.api import ApiRequest, ApiRequestV2
from aiounifi.models.dpi_restriction_app import DPIRestrictionApp  # Import DPIApp model
from aiounifi.models.dpi_restriction_group import (
    DPIRestrictionGroup,
)  # Import DPIGroup model

from unifi_core.mac import mac_equal, normalize_mac
from unifi_core.network.managers.client_manager import ClientManager  # Needed for get_top_clients
from unifi_core.network.managers.connection_manager import ConnectionManager
from unifi_core.network.managers.device_manager import DeviceManager
from unifi_core.network.system_log import (
    EVENT_SEVERITIES,
    SYSTEM_LOG_PAGE_SIZE,
    PageTracker,
    SystemLogIncompleteError,
    SystemLogResponseError,
    coverage,
    describe_request_error,
    is_endpoint_absent,
    parse_system_log_page,
)

logger = logging.getLogger("unifi-network-mcp")

#: Bound on system-log pages read for one IPS query. Beyond it the result is
#: reported as truncated rather than fetched without limit.
IPS_SYSTEM_LOG_MAX_PAGES = 10
IPS_SYSTEM_LOG_MAX_EVENTS = IPS_SYSTEM_LOG_MAX_PAGES * SYSTEM_LOG_PAGE_SIZE

# Cache prefixes
CACHE_PREFIX_STATS_NETWORK = "stats_network"
CACHE_PREFIX_STATS_CLIENT = "stats_client"
CACHE_PREFIX_STATS_DEVICE = "stats_device"
CACHE_PREFIX_STATS_DPI = "stats_dpi"
CACHE_PREFIX_STATS_ALERTS = "stats_alerts"
CACHE_PREFIX_STATS_GATEWAY = "stats_gateway"
CACHE_PREFIX_STATS_SPEEDTEST = "stats_speedtest"
CACHE_PREFIX_STATS_SITE_DPI = "stats_site_dpi"
CACHE_PREFIX_STATS_CLIENT_DPI = "stats_client_dpi"
CACHE_PREFIX_STATS_IPS = "stats_ips"
CACHE_PREFIX_STATS_SESSIONS = "stats_sessions"
CACHE_PREFIX_STATS_DASHBOARD = "stats_dashboard"
CACHE_PREFIX_STATS_ANOMALIES = "stats_anomalies"
CACHE_PREFIX_STATS_CLIENT_WIFI = "stats_client_wifi"

# Granularity validation
VALID_GRANULARITIES = {"5minutes", "hourly", "daily", "monthly"}


def _resolve_granularity(granularity: str) -> str:
    """Validate and return the granularity string.

    Args:
        granularity: One of 5minutes, hourly, daily, monthly.

    Returns:
        The validated granularity string.

    Raises:
        ValueError: If the granularity is not valid.
    """
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(
            f"Invalid granularity '{granularity}'. Must be one of: {', '.join(sorted(VALID_GRANULARITIES))}"
        )
    return granularity


class StatsManager:
    """Manages statistics retrieval from the Unifi Controller."""

    def __init__(self, connection_manager: ConnectionManager, client_manager: ClientManager):
        """Initialize the Stats Manager.

        Args:
            connection_manager: The shared ConnectionManager instance.
            client_manager: The ClientManager instance (needed for some stats methods).
        """
        self._connection = connection_manager
        self._client_manager = client_manager

    async def get_network_stats(self, duration_hours: int = 1, granularity: str = "hourly") -> List[Dict[str, Any]]:
        """Get network statistics (e.g., hourly site stats)."""
        granularity = _resolve_granularity(granularity)
        cache_key = f"{CACHE_PREFIX_STATS_NETWORK}_{duration_hours}_{granularity}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return cached_data

        try:
            start_time = int((datetime.now() - timedelta(hours=duration_hours)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)

            endpoint = f"/stat/report/{granularity}.site"
            # Use non-rate attributes commonly available on report endpoints
            payload = {
                "attrs": [
                    "bytes",  # total bytes (if provided by controller)
                    "rx_bytes",  # some controllers provide rx/tx at site level
                    "tx_bytes",
                    "num_user",
                    "num_sta",
                    "num_active_user",
                    "wan-rx_bytes",
                    "wan-tx_bytes",
                    "wlan-num_sta",
                    "lan-num_sta",
                ],
                "start": start_time,
                "end": end_time,
            }
            api_request = ApiRequest(method="post", path=endpoint, data=payload)
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=300)
            return result
        except Exception as e:
            logger.error("Error getting network stats: %s", e)
            raise

    async def get_client_stats(
        self, client_mac: str, duration_hours: int = 1, granularity: str = "hourly"
    ) -> List[Dict[str, Any]]:
        """Get statistics for a specific client."""
        client_mac = normalize_mac(client_mac) or client_mac
        granularity = _resolve_granularity(granularity)
        cache_key = f"{CACHE_PREFIX_STATS_CLIENT}_{client_mac}_{duration_hours}_{granularity}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return cached_data

        try:
            start_time = int((datetime.now() - timedelta(hours=duration_hours)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)

            endpoint = f"/stat/report/{granularity}.user"
            payload = {
                "attrs": [
                    "rx_bytes",
                    "tx_bytes",
                    "bytes",
                    "signal",
                    "satisfaction",
                    "tx_retries",
                    "tx_rate",
                    "rx_rate",
                    "wifi_tx_attempts",
                ],
                "mac": client_mac,
                "start": start_time,
                "end": end_time,
            }
            api_request = ApiRequest(method="post", path=endpoint, data=payload)
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=300)
            return result
        except Exception as e:
            logger.error("Error getting stats for client %s: %s", client_mac, e)
            raise

    async def get_client_stats_for_identifier(
        self,
        client_id: str,
        duration_hours: int = 1,
        granularity: str = "hourly",
    ) -> List[Dict[str, Any]]:
        """Resolve a public MAC-or-id client identifier before fetching stats."""
        client = await self._client_manager.get_client_details(client_id)
        raw = client.raw if hasattr(client, "raw") else client
        client_mac = raw.get("mac", client_id)
        return await self.get_client_stats(
            client_mac,
            duration_hours=duration_hours,
            granularity=granularity,
        )

    async def get_device_stats(
        self,
        device_mac: str,
        duration_hours: int = 1,
        granularity: str = "hourly",
        device_type: str = "dev",
    ) -> List[Dict[str, Any]]:
        """Get statistics for a specific device.

        Args:
            device_mac: MAC address of the device.
            duration_hours: Number of hours to look back.
            granularity: Report granularity (5minutes, hourly, daily, monthly).
            device_type: Device type for endpoint routing (ap, gw, or dev).
        """
        device_mac = normalize_mac(device_mac) or device_mac
        granularity = _resolve_granularity(granularity)
        cache_key = (
            f"{CACHE_PREFIX_STATS_DEVICE}_{device_mac}_{duration_hours}_{granularity}"
            f"_{device_type}_{self._connection.site}"
        )
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return cached_data

        try:
            start_time = int((datetime.now() - timedelta(hours=duration_hours)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)

            # Route endpoint and attrs based on device_type
            if device_type == "ap":
                endpoint = f"/stat/report/{granularity}.ap"
                attrs = [
                    "num_sta",
                    "ng-num_sta",
                    "na-num_sta",
                    "satisfaction",
                    "tx_retries",
                    "wifi_tx_attempts",
                    "wifi_tx_dropped",
                    "ng-tx_bytes",
                    "na-tx_bytes",
                    "ng-rx_bytes",
                    "na-rx_bytes",
                ]
            elif device_type == "gw":
                endpoint = f"/stat/report/{granularity}.gw"
                attrs = [
                    "wan-rx_bytes",
                    "wan-tx_bytes",
                    "lan-rx_bytes",
                    "lan-tx_bytes",
                    "cpu",
                    "mem",
                    "loadavg_5",
                ]
            else:
                endpoint = f"/stat/report/{granularity}.dev"
                attrs = [
                    "rx_bytes",
                    "tx_bytes",
                    "bytes",
                    "num_sta",
                ]

            payload = {
                "attrs": attrs,
                "mac": device_mac,
                "start": start_time,
                "end": end_time,
            }
            api_request = ApiRequest(method="post", path=endpoint, data=payload)
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=300)
            return result
        except Exception as e:
            logger.error("Error getting stats for device %s: %s", device_mac, e)
            raise

    async def get_device_stats_for_identifier(
        self,
        device_id: str,
        duration_hours: int = 1,
        granularity: str = "hourly",
    ) -> List[Dict[str, Any]]:
        """Resolve a public MAC-or-id device identifier and select its report family."""
        device = await DeviceManager(self._connection).get_device_details(device_id)
        raw = device.raw if hasattr(device, "raw") else device
        device_mac = raw.get("mac", device_id)
        device_type = raw.get("type", "unknown")
        report_type = {"uap": "ap", "ugw": "gw", "udm": "gw"}.get(device_type, "dev")
        return await self.get_device_stats(
            device_mac,
            duration_hours=duration_hours,
            granularity=granularity,
            device_type=report_type,
        )

    async def get_top_clients(self, duration_hours: int = 24, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top clients by usage.

        Fallback approach: derive usage from current client objects when report
        endpoints return no data. Values are cumulative since association and
        are sufficient for ranking purposes.
        """
        online_clients = await self._client_manager.get_clients()
        if not online_clients:
            return []

        clients_raw = [c.raw if hasattr(c, "raw") else c for c in online_clients]

        aggregated_stats: List[Dict[str, Any]] = []

        def _safe_int(value: Any) -> int:
            try:
                if value is None:
                    return 0
                if isinstance(value, bool):
                    return int(value)
                return int(str(value))
            except Exception:
                return 0

        for client in clients_raw:
            mac = client.get("mac")
            if not mac:
                continue

            # Prefer explicit total if present, otherwise sum rx/tx (wifi + wired)
            rx = _safe_int(client.get("rx_bytes")) + _safe_int(
                client.get("wired_rx_bytes") or client.get("wired-rx_bytes")
            )
            tx = _safe_int(client.get("tx_bytes")) + _safe_int(
                client.get("wired_tx_bytes") or client.get("wired-tx_bytes")
            )
            total = _safe_int(client.get("bytes")) or (rx + tx)

            aggregated_stats.append(
                {
                    "mac": mac,
                    "name": client.get("name") or client.get("hostname", mac),
                    "rx_bytes": rx,
                    "tx_bytes": tx,
                    "total_bytes": total,
                }
            )

        # Sort by total bytes descending and return top N
        sorted_clients = sorted(aggregated_stats, key=lambda x: x["total_bytes"], reverse=True)
        return sorted_clients[:limit]

    async def get_dpi_stats(
        self,
    ) -> Dict[str, List[Any]]:  # Return List[DPIRestrictionApp/Group]
        """Get Deep Packet Inspection (DPI) statistics."""
        cache_key = f"{CACHE_PREFIX_STATS_DPI}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=900)  # 15 minute cache
        if cached_data is not None:
            return cached_data

        if not await self._connection.ensure_connected() or not self._connection.controller:
            raise ConnectionError("Not connected to controller")

        try:
            await self._connection.refresh_handler("dpi_apps")
            await self._connection.refresh_handler("dpi_groups")

            dpi_apps: List[DPIRestrictionApp] = list(self._connection.controller.dpi_apps.values())
            dpi_groups: List[DPIRestrictionGroup] = list(self._connection.controller.dpi_groups.values())
            result = {"applications": dpi_apps, "categories": dpi_groups}
            self._connection._update_cache(cache_key, result, timeout=900)
            return result
        except Exception as e:
            logger.error("Error getting DPI stats: %s", e)
            raise

    async def get_alerts(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Get alerts from the controller."""
        cache_key = f"{CACHE_PREFIX_STATS_ALERTS}_{include_archived}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=60)  # 1 minute cache
        if cached_data is not None:
            return cached_data

        if not await self._connection.ensure_connected() or not self._connection.controller:
            raise ConnectionError("Not connected to controller")
        try:
            alerts = await self._get_alerts_v2()
        except Exception as v2_error:
            logger.debug("V2 alert lookup failed, falling back to legacy alarms: %s", v2_error)
            try:
                alerts = await self._get_alerts_legacy(include_archived)
            except Exception as legacy_error:
                # Report both. Current controllers have removed /stat/alarm, so the
                # legacy error is usually a 404 that says nothing about the real
                # problem — which is whatever made the v2 lookup fail.
                raise RuntimeError(
                    f"/stat/alarm failed ({legacy_error}) after the v2 system-log lookup failed "
                    f"({v2_error}). On current UniFi Network versions the legacy endpoint no longer "
                    "exists, so the v2 failure is the one to act on."
                ) from legacy_error

        if not include_archived:
            alerts = [a for a in alerts if not a.get("archived", False)]
        self._connection._update_cache(cache_key, alerts, timeout=60)
        return alerts

    async def _get_alerts_v2(self) -> List[Dict[str, Any]]:
        """Get alerts from the v2 system-log critical endpoint."""
        now_ms = int(datetime.now().timestamp() * 1000)
        from_ms = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
        api_request = ApiRequestV2(
            method="post",
            path="/system-log/critical",
            data={
                "timestampFrom": from_ms,
                "timestampTo": now_ms,
                "severities": ["HIGH", "VERY_HIGH"],
                "categories": [
                    "CLIENT_DEVICES",
                    "INTERNET_AND_WAN",
                    "POWER",
                    "SECURITY",
                    "UNIFI_DEVICES",
                    "SOFTWARE_UPDATES",
                    "UNIFI_ETHERNET_PORTS",
                    "VPN",
                ],
                "type": "GENERAL",
                "pageNumber": 0,
                "pageSize": 100,
                "searchText": "",
            },
        )
        response = await self._connection.request(api_request)
        if isinstance(response, list) and response and isinstance(response[0], dict) and "data" in response[0]:
            return response[0]["data"]
        if isinstance(response, dict):
            data = response.get("data", response.get("logs", []))
            return data if isinstance(data, list) else []
        if isinstance(response, list):
            return response
        return []

    async def _get_alerts_legacy(self, include_archived: bool) -> List[Dict[str, Any]]:
        """Get alerts from the legacy alarm endpoint."""
        path = "/stat/alarm?archived=true" if include_archived else "/stat/alarm"
        api_request = ApiRequest(method="get", path=path)
        response = await self._connection.request(api_request)
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            data = response.get("data", [])
            return data if isinstance(data, list) else []
        return []

    async def get_gateway_stats(self, duration_hours: int = 24, granularity: str = "hourly") -> List[Dict[str, Any]]:
        """Get gateway statistics."""
        granularity = _resolve_granularity(granularity)
        cache_key = f"{CACHE_PREFIX_STATS_GATEWAY}_{duration_hours}_{granularity}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return cached_data

        try:
            start_time = int((datetime.now() - timedelta(hours=duration_hours)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)

            endpoint = f"/stat/report/{granularity}.gw"
            payload = {
                "attrs": [
                    "wan-rx_bytes",
                    "wan-tx_bytes",
                    "lan-rx_bytes",
                    "lan-tx_bytes",
                    "cpu",
                    "mem",
                    "loadavg_5",
                    "wan-rx_packets",
                    "wan-tx_packets",
                ],
                "start": start_time,
                "end": end_time,
            }
            api_request = ApiRequest(method="post", path=endpoint, data=payload)
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=300)
            return result
        except Exception as e:
            logger.error("Error getting gateway stats: %s", e)
            raise

    async def get_speedtest_results(self, duration_hours: int = 24) -> List[Dict[str, Any]]:
        """Get speed test results from the controller archive."""
        cache_key = f"{CACHE_PREFIX_STATS_SPEEDTEST}_{duration_hours}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return cached_data

        try:
            start_time = int((datetime.now() - timedelta(hours=duration_hours)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)

            endpoint = "/stat/report/archive.speedtest"
            payload = {
                "attrs": [
                    "xput_download",
                    "xput_upload",
                    "latency",
                    "time",
                ],
                "start": start_time,
                "end": end_time,
            }
            api_request = ApiRequest(method="post", path=endpoint, data=payload)
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=300)
            return result
        except Exception as e:
            logger.error("Error getting speedtest results: %s", e)
            raise

    async def get_site_dpi_traffic(self, by: str = "by_app") -> List[Dict[str, Any]]:
        """Get site-level DPI traffic statistics.

        Args:
            by: Grouping type, e.g. 'by_app' or 'by_cat'.
        """
        cache_key = f"{CACHE_PREFIX_STATS_SITE_DPI}_{by}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=900)  # 15 minute cache
        if cached_data is not None:
            return cached_data

        try:
            endpoint = "/stat/sitedpi"
            payload = {"type": by}
            api_request = ApiRequest(method="post", path=endpoint, data=payload)
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=900)
            return result
        except Exception as e:
            logger.error("Error getting site DPI traffic: %s", e)
            raise

    async def get_client_dpi_traffic(self, client_mac: str, by: str = "by_app") -> List[Dict[str, Any]]:
        """Get DPI traffic statistics for a specific client.

        Args:
            client_mac: MAC address of the client.
            by: Grouping type, e.g. 'by_app' or 'by_cat'.
        """
        client_mac = normalize_mac(client_mac) or client_mac
        cache_key = f"{CACHE_PREFIX_STATS_CLIENT_DPI}_{client_mac}_{by}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=900)  # 15 minute cache
        if cached_data is not None:
            return cached_data

        try:
            endpoint = "/stat/stadpi"
            payload = {"type": by, "macs": [client_mac]}
            api_request = ApiRequest(method="post", path=endpoint, data=payload)
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=900)
            return result
        except Exception as e:
            logger.error("Error getting DPI traffic for client %s: %s", client_mac, e)
            raise

    async def get_ips_events(self, duration_hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Get IPS/IDS events as a plain list.

        See ``get_ips_event_report`` for the source and coverage metadata. When
        an empty legacy answer could not be cross-checked against the system log
        (``system_log_cross_check="failed"``), this deliberately still returns
        the legacy answer (``[]``); the MCP tool fails closed in that case, so
        callers that need a confirmed window should use the report.

        Args:
            duration_hours: Number of hours to look back.
            limit: Maximum number of events to return.
        """
        report = await self.get_ips_event_report(duration_hours=duration_hours, limit=limit)
        return report["events"]

    async def get_ips_event_report(self, duration_hours: int = 24, limit: int = 50) -> Dict[str, Any]:
        """Get IPS/IDS events with the source that answered and how complete the answer is.

        The legacy ``/stat/ips/event`` endpoint is asked first and its non-empty
        answer is used as-is. When it is unavailable, or answers with nothing,
        the v2 system log is queried for SECURITY events at every severity.

        - legacy unavailable, system log answers: ``source="system-log"`` with
          the controller's total and the coverage flags.
        - legacy unavailable, system log fails, answers in an unreadable or
          self-contradictory shape, or answers empty without an exact total
          (none, or only a lower bound): raises.
          Nothing is cached.
        - legacy answers empty and the system log is absent (``/system-log/all``
          and the ``/system-log/count`` probe both 404/405): the legacy answer
          stands with ``system_log_cross_check="absent"``; controllers that
          predate the system log answer exactly this way. Not cached, since a
          404 can also come from a proxy or path problem.
        - legacy answers empty and the system log exists but fails (including a
          failure after earlier pages were read): the legacy answer is returned
          uncached with ``system_log_cross_check="failed"`` and the error.
          Callers that need a confirmed answer must treat this as a failure.

        Raises:
            ValueError: when ``limit`` is below 1.

        Returns:
            Dict with ``events``, ``source``, ``window``, ``total``, ``complete``,
            ``truncated``, ``max_events`` and, for legacy answers,
            ``system_log_cross_check`` (``"skipped"``, ``"absent"`` or ``"failed"``).
            A copy, so callers may modify it without touching the cache.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit must be an integer of at least 1, got {limit!r}")

        cache_key = f"{CACHE_PREFIX_STATS_IPS}_{duration_hours}_{limit}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return copy.deepcopy(cached_data)

        try:
            start_time = int((datetime.now() - timedelta(hours=duration_hours)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)
            window = {"duration_hours": duration_hours, "start_ms": start_time, "end_ms": end_time}

            def legacy_report(events: List[Dict[str, Any]], cross_check: str) -> Dict[str, Any]:
                return {
                    "events": events,
                    "source": "legacy",
                    "window": window,
                    # The legacy endpoint reports no total, so coverage is unknown.
                    "total": None,
                    "complete": None,
                    "truncated": None,
                    "max_events": limit,
                    "system_log_cross_check": cross_check,
                }

            api_request = ApiRequest(
                method="post",
                path="/stat/ips/event",
                data={
                    "start": start_time,
                    "end": end_time,
                    "_limit": limit,
                },
            )
            try:
                response = await self._connection.request(api_request)
            except Exception as legacy_error:
                logger.debug("Legacy IPS endpoint failed, falling back to the v2 system log: %s", legacy_error)
                try:
                    report = await self._get_security_events_v2(start_time, end_time, limit)
                except Exception as v2_error:
                    raise RuntimeError(
                        f"the legacy IPS endpoint /stat/ips/event failed ({describe_request_error(legacy_error)}) "
                        f"and the system-log fallback /system-log/all failed ({describe_request_error(v2_error)}); "
                        "no source answered, so the absence of events cannot be confirmed"
                    ) from v2_error
                report["window"] = window
            else:
                if isinstance(response, list) and response:
                    report = legacy_report(response, "skipped")
                else:
                    # A controller that has moved IPS events to the v2 system log answers
                    # this endpoint 200 with an empty list rather than failing, so an
                    # empty legacy answer is cross-checked against the system log.
                    logger.debug("Legacy IPS endpoint returned no events, trying the v2 system log")
                    try:
                        report = await self._get_security_events_v2(start_time, end_time, limit)
                        report["window"] = window
                    except Exception as v2_error:
                        if not isinstance(response, list):
                            raise RuntimeError(
                                "the legacy IPS endpoint /stat/ips/event answered in an unreadable shape "
                                f"({type(response).__name__}) and the system-log fallback /system-log/all "
                                f"failed ({describe_request_error(v2_error)})"
                            ) from v2_error
                        failure = await self._system_log_failure(v2_error, start_time, end_time)
                        if failure is None:
                            # A controller that predates the system log: the empty legacy
                            # answer is the authoritative one. Uncached, because a 404 can
                            # also be a path or proxy problem that clears on retry.
                            return legacy_report([], "absent")
                        # Deliberately not raising: legacy did answer, and the caller
                        # decides. Uncached, so a transient outage is retried next call
                        # rather than held as "no threats".
                        logger.warning("System-log cross-check of an empty IPS answer failed: %s", failure)
                        report = legacy_report([], "failed")
                        report["system_log_cross_check_error"] = failure
                        return report
            self._connection._update_cache(cache_key, copy.deepcopy(report), timeout=300)
            return report
        except Exception as e:
            logger.error("Error getting IPS events: %s", e)
            raise

    async def _system_log_failure(self, error: Exception, start_time: int, end_time: int) -> Optional[str]:
        """Classify a system-log error after an empty legacy answer.

        Returns None when the system log is absent on this controller, which
        takes a 404/405 on the first ``/system-log/all`` request and on the
        ``/system-log/count`` probe used for API detection. Otherwise returns a
        short description of the failure.
        """
        description = describe_request_error(error)
        if isinstance(error, SystemLogIncompleteError) or not is_endpoint_absent(error):
            return description
        probe = ApiRequestV2(
            method="post",
            path="/system-log/count",
            data={
                "timestampFrom": start_time,
                "timestampTo": end_time,
                "severities": list(EVENT_SEVERITIES),
                "categories": ["SECURITY"],
                "type": "GENERAL",
            },
        )
        try:
            await self._connection.request(probe)
        except Exception as probe_error:
            if is_endpoint_absent(probe_error):
                return None
            return f"{description}; the /system-log/count probe also failed ({describe_request_error(probe_error)})"
        return f"{description}, although /system-log/count answered, so the system log exists"

    async def _get_security_events_v2(self, start_time: int, end_time: int, limit: int) -> Dict[str, Any]:
        """Read SECURITY events at every severity from ``/system-log/all``.

        Pages through the window until ``limit`` events are collected, the
        controller runs out, or ``IPS_SYSTEM_LOG_MAX_PAGES`` pages have been
        read. A failure of the first request propagates as-is; a failure after
        a page was read raises ``SystemLogIncompleteError``. Unreadable
        responses, an empty page that contradicts the counts, and an empty
        answer without an exact total raise ``SystemLogResponseError``; only an
        exact total of 0 that is not a lower bound confirms a quiet window.
        """
        max_events = min(limit, IPS_SYSTEM_LOG_MAX_EVENTS)
        tracker = PageTracker(SYSTEM_LOG_PAGE_SIZE)
        events: List[Dict[str, Any]] = []
        page_number = 0
        more = True
        while more and len(events) < max_events and page_number < IPS_SYSTEM_LOG_MAX_PAGES:
            api_request = ApiRequestV2(
                method="post",
                path="/system-log/all",
                data={
                    "timestampFrom": start_time,
                    "timestampTo": end_time,
                    "severities": list(EVENT_SEVERITIES),
                    "categories": ["SECURITY"],
                    "type": "GENERAL",
                    "pageNumber": page_number,
                    "pageSize": SYSTEM_LOG_PAGE_SIZE,
                    "searchText": "",
                },
            )
            try:
                page = parse_system_log_page(await self._connection.request(api_request))
                tracker.observe(page)
                more = tracker.has_next_page(page, page_number)
            except Exception as page_error:
                if page_number == 0:
                    raise
                raise SystemLogIncompleteError(
                    f"system-log page {page_number + 1} failed after {len(events)} events were read "
                    f"({describe_request_error(page_error)})"
                ) from page_error
            events.extend(page.events)
            page_number += 1

        if not events and (tracker.total != 0 or tracker.total_is_lower_bound):
            # Only an exact total of 0 confirms a quiet window; no count, or a
            # lower-bound count ("0 or more"), confirms nothing.
            raise SystemLogResponseError(
                "the system log returned no events and no exact total, so an empty window is unconfirmed"
            )
        stopped_early = more or len(events) > max_events
        events = events[:max_events]
        return {
            "events": events,
            "source": "system-log",
            "total": tracker.total,
            **coverage(len(events), tracker.total, tracker.total_is_lower_bound, stopped_early),
            "max_events": max_events,
        }

    async def get_client_sessions(
        self,
        client_mac: Optional[str] = None,
        duration_hours: int = 24,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get client session history.

        Args:
            client_mac: Optional MAC address to filter sessions for a specific client.
            duration_hours: Number of hours to look back.
            limit: Maximum number of sessions to return.
        """
        client_mac = normalize_mac(client_mac) or client_mac
        mac_part = client_mac or "all"
        cache_key = f"{CACHE_PREFIX_STATS_SESSIONS}_{mac_part}_{duration_hours}_{limit}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return cached_data

        try:
            start_time = int((datetime.now() - timedelta(hours=duration_hours)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)

            endpoint = "/stat/session"
            payload: Dict[str, Any] = {
                "type": "all",
                "start": start_time,
                "end": end_time,
                "_limit": limit,
            }
            if client_mac:
                payload["mac"] = client_mac

            api_request = ApiRequest(method="post", path=endpoint, data=payload)
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=300)
            return result
        except Exception as e:
            logger.error("Error getting client sessions: %s", e)
            raise

    async def get_dashboard(self, history_seconds: int = 86400) -> List[Dict[str, Any]]:
        """Get the site dashboard aggregate (health, WiFi standards, ISP metrics).

        Hits the V2 ``/aggregated-dashboard`` endpoint that the UniFi UI uses
        to render its dashboard view. The response is a list of one or more
        dashboard payloads (typically one per controller-managed site rollup).
        Each payload contains nested blocks like ``wifi`` (per-standard
        connection counts and traffic), ``isp_metrics`` (per-WAN history),
        and other UI-visible aggregates.

        Args:
            history_seconds: How far back to roll up history. Defaults to
                86400 (24h), matching the UI default. The endpoint requires
                this query parameter.
        """
        cache_key = f"{CACHE_PREFIX_STATS_DASHBOARD}_{history_seconds}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return cached_data

        try:
            api_request = ApiRequestV2(
                method="get",
                path=f"/aggregated-dashboard?historySeconds={int(history_seconds)}",
            )
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=300)
            return result
        except Exception as e:
            logger.error("Error getting dashboard: %s", e)
            raise

    async def get_anomalies(self, duration_hours: int = 24) -> List[Dict[str, Any]]:
        """Get detected anomalies.

        Args:
            duration_hours: Number of hours to look back.
        """
        cache_key = f"{CACHE_PREFIX_STATS_ANOMALIES}_{duration_hours}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return cached_data

        try:
            start_time = int((datetime.now() - timedelta(hours=duration_hours)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)

            endpoint = "/stat/anomalies"
            payload = {
                "start": start_time,
                "end": end_time,
            }
            api_request = ApiRequest(method="post", path=endpoint, data=payload)
            response = await self._connection.request(api_request)
            result = response if isinstance(response, list) else []
            self._connection._update_cache(cache_key, result, timeout=300)
            return result
        except Exception as e:
            logger.error("Error getting anomalies: %s", e)
            raise

    async def get_client_wifi_details(self, client_mac: str) -> Optional[Dict[str, Any]]:
        """Get detailed WiFi information for a specific client.

        Args:
            client_mac: MAC address of the client.

        Returns:
            Dict with WiFi detail fields, or None if client not found.
        """
        client_mac = normalize_mac(client_mac) or client_mac
        cache_key = f"{CACHE_PREFIX_STATS_CLIENT_WIFI}_{client_mac}_{self._connection.site}"
        cached_data = self._connection.get_cached(cache_key, timeout=300)  # 5 minute cache
        if cached_data is not None:
            return cached_data

        try:
            # /stat/sta is a collection endpoint: it returns all currently-
            # connected wireless clients and ignores any MAC filter in the body.
            # Fetch the full list, then filter for the requested MAC ourselves.
            api_request = ApiRequest(method="get", path="/stat/sta")
            response = await self._connection.request(api_request)

            clients = response if isinstance(response, list) else []
            raw = next(
                (c for c in clients if isinstance(c, dict) and mac_equal(c.get("mac"), client_mac)),
                None,
            )
            if raw is None:
                return None

            wifi_fields = [
                "signal",
                "noise",
                "satisfaction",
                "tx_rate",
                "rx_rate",
                "tx_retries",
                "wifi_tx_attempts",
                "roam_count",
                "os_name",
                "dev_vendor",
                "dev_cat",
                "dev_family",
                "channel",
                "radio",
                "essid",
                "bssid",
                "nss",
                "is_11r",
            ]
            result: Dict[str, Any] = {"mac": raw.get("mac", client_mac)}
            for field in wifi_fields:
                if field in raw:
                    result[field] = raw[field]

            self._connection._update_cache(cache_key, result, timeout=300)
            return result
        except Exception as e:
            logger.error("Error getting WiFi details for client %s: %s", client_mac, e)
            raise
