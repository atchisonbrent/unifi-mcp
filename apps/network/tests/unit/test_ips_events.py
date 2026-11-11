"""IPS event retrieval through ``unifi_get_ips_events`` and the real StatsManager.

The controller is mocked at ``ConnectionManager.request`` and routed by path,
so the manager's real fallback, paging, and coverage logic runs. Fixtures are
synthetic (documentation address ranges, invented names).
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiounifi.errors import AiounifiException, ResponseError

os.environ.setdefault("UNIFI_HOST", "127.0.0.1")
os.environ.setdefault("UNIFI_USERNAME", "test")
os.environ.setdefault("UNIFI_PASSWORD", "test")

LEGACY = "/stat/ips/event"
SYSTEM_LOG_ALL = "/system-log/all"
SYSTEM_LOG_CRITICAL = "/system-log/critical"
SYSTEM_LOG_COUNT = "/system-log/count"
CONTROLLER_HOST = "192.0.2.1"


def _not_found(path: str) -> ResponseError:
    return ResponseError(f"Call https://192.0.2.1/proxy/network{path} received 404 Not Found")


def _server_error() -> AiounifiException:
    return AiounifiException({"errorCode": 500, "message": "internal error"})


def _threat(index: int, key: str = "THREAT_BLOCKED_KNOWN_SOURCE_CLIENT", severity: str = "LOW") -> dict:
    return {
        "id": f"threat-{index:04d}",
        "key": key,
        "category": "SECURITY",
        "subcategory": "INTRUSION_PREVENTION",
        "severity": severity,
        "timestamp": 1786000000000 - index * 1000,
        "message_raw": "{SRC_CLIENT} was blocked from reaching {DST_IP}.",
        "parameters": {
            "SRC_CLIENT": {"id": "02:00:00:00:00:01", "ip": "192.0.2.21", "name": "Test-Laptop"},
            "DST_IP": {"id": "198.51.100.7", "name": "198.51.100.7"},
        },
    }


def _envelope(events: list, total: int | None, page_number: int = 0, page_size: int = 100, **extra) -> list:
    envelope = {"data": events, "page_number": page_number, **extra}
    if total is not None:
        envelope["total_element_count"] = total
        envelope["total_page_count"] = -(-total // page_size) if total else 0
    return [envelope]


def _paged(all_events: list, total: int | None = None) -> callable:
    """Serve ``all_events`` from /system-log/all in controller-sized pages."""
    total = len(all_events) if total is None else total

    def respond(data: dict) -> list:
        page_number, page_size = data["pageNumber"], data["pageSize"]
        start = page_number * page_size
        return _envelope(all_events[start : start + page_size], total, page_number, page_size)

    return respond


class _Controller:
    """Routes requests by path; records every request made."""

    def __init__(self, legacy, system_log_all, system_log_critical=None, system_log_count=None):
        self.legacy = legacy
        self.system_log_all = system_log_all
        # The /system-log/critical view answers empty for the same window, as
        # observed on current controllers; a fix that still reads it fails.
        self.system_log_critical = system_log_critical or _envelope([], 0)
        self.system_log_count = system_log_count if system_log_count is not None else [{"count": 0}]
        self.requests = []

    async def __call__(self, api_request):
        self.requests.append(api_request)
        handler = {
            LEGACY: self.legacy,
            SYSTEM_LOG_ALL: self.system_log_all,
            SYSTEM_LOG_CRITICAL: self.system_log_critical,
            SYSTEM_LOG_COUNT: self.system_log_count,
        }[api_request.path]
        if isinstance(handler, BaseException):
            raise handler
        if callable(handler):
            return handler(api_request.data)
        return handler

    def paths(self) -> list:
        return [request.path for request in self.requests]


def _connection(controller: _Controller, cache: dict | None = None) -> MagicMock:
    """Mock connection; pass ``cache`` to back get_cached/_update_cache with a real dict."""
    conn = MagicMock()
    conn.site = "default"
    conn.request = AsyncMock(side_effect=controller.__call__)
    if cache is None:
        conn.get_cached = MagicMock(return_value=None)
        conn._update_cache = MagicMock()
    else:
        conn.get_cached = MagicMock(side_effect=lambda key, timeout=None: cache.get(key))
        conn._update_cache = MagicMock(side_effect=lambda key, value, timeout=None: cache.__setitem__(key, value))
    return conn


def _fail_on_page(events: list, failing_page: int, error: BaseException, total: int) -> callable:
    """Serve pages of ``events`` until ``failing_page``, which raises ``error``."""
    serve = _paged(events, total)

    def respond(data: dict) -> list:
        if data["pageNumber"] == failing_page:
            raise error
        return serve(data)

    return respond


def _assert_no_controller_address(text: str) -> None:
    assert CONTROLLER_HOST not in text
    assert "://" not in text
    assert "unifi.example" not in text
    assert "unifi:8443" not in text


def _manager(conn):
    from unifi_core.network.managers.stats_manager import StatsManager

    return StatsManager(conn, MagicMock())


async def _call_tool(controller: _Controller, **kwargs):
    conn = _connection(controller)
    manager = _manager(conn)
    with patch("unifi_network_mcp.tools.stats.stats_manager", manager):
        from unifi_network_mcp.tools.stats import get_ips_events

        return await get_ips_events(**kwargs), conn


# ---------------------------------------------------------------------------
# Legacy unavailable, system log answers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_404_falls_back_to_system_log_and_returns_the_threat():
    """The false all-clear: legacy 404, /system-log/critical empty, /system-log/all holds the threat."""
    threat = _threat(1, key="THREAT_BLOCKED_KNOWN_DESTINATION_CLIENT", severity="MEDIUM")
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=_envelope([threat], 1))

    result, _ = await _call_tool(controller)

    assert result["success"] is True
    assert result["source"] == "system-log"
    assert result["count"] == 1
    event = result["events"][0]
    assert event["key"] == "THREAT_BLOCKED_KNOWN_DESTINATION_CLIENT"
    assert event["category"] == "SECURITY"
    assert result["total"] == 1
    assert result["complete"] is True
    assert result["truncated"] is False
    assert controller.paths() == [LEGACY, SYSTEM_LOG_ALL]
    fallback = controller.requests[1].data
    assert fallback["categories"] == ["SECURITY"]
    assert fallback["severities"] == ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    window = result["window"]
    assert window["duration_hours"] == 24
    assert (fallback["timestampFrom"], fallback["timestampTo"]) == (window["start_ms"], window["end_ms"])


@pytest.mark.asyncio
async def test_material_threat_beyond_the_first_page_is_collected():
    """101 events, the one that matters is #101. Stopping at page one must fail this test."""
    routine = [_threat(i) for i in range(100)]
    material = _threat(100, key="THREAT_BLOCKED_KNOWN_SOURCE_AND_DESTINATION_CLIENTS", severity="VERY_HIGH")
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=_paged(routine + [material]))

    result, _ = await _call_tool(controller, limit=200)

    assert result["success"] is True
    assert result["count"] == 101
    assert result["total"] == 101
    assert result["complete"] is True
    assert result["truncated"] is False
    assert result["events"][-1]["key"] == "THREAT_BLOCKED_KNOWN_SOURCE_AND_DESTINATION_CLIENTS"
    assert [request.data["pageNumber"] for request in controller.requests[1:]] == [0, 1]


@pytest.mark.parametrize(
    ("available", "limit", "expected_count", "complete", "truncated"),
    [
        (0, 50, 0, True, False),
        (12, 50, 12, True, False),
        (101, 100, 100, False, True),
    ],
    ids=["0-of-0", "12-of-12", "100-of-101"],
)
@pytest.mark.asyncio
async def test_coverage_flags_follow_collected_versus_total(available, limit, expected_count, complete, truncated):
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=_paged([_threat(i) for i in range(available)]))

    result, _ = await _call_tool(controller, limit=limit)

    assert result["success"] is True
    assert result["source"] == "system-log"
    assert result["count"] == expected_count
    assert result["total"] == available
    assert result["complete"] is complete
    assert result["truncated"] is truncated


@pytest.mark.asyncio
async def test_genuinely_empty_system_log_is_a_successful_zero():
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=_envelope([], 0))

    result, conn = await _call_tool(controller)

    assert result["success"] is True
    assert result["count"] == 0
    assert result["source"] == "system-log"
    assert result["complete"] is True
    conn._update_cache.assert_called_once()


@pytest.mark.asyncio
async def test_missing_total_leaves_coverage_unknown():
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=[_threat(1), _threat(2)])

    result, _ = await _call_tool(controller)

    assert result["success"] is True
    assert result["count"] == 2
    assert result["total"] is None
    assert result["complete"] is None
    assert result["truncated"] is None


@pytest.mark.asyncio
async def test_lower_bound_total_does_not_claim_completeness():
    events = [_threat(i) for i in range(12)]
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=_envelope(events, 12, or_more=True))

    result, _ = await _call_tool(controller)

    assert result["count"] == 12
    assert result["complete"] is None
    assert result["truncated"] is None


@pytest.mark.asyncio
async def test_page_cap_bounds_the_fetch_and_reports_truncation():
    from unifi_core.network.managers.stats_manager import IPS_SYSTEM_LOG_MAX_EVENTS, IPS_SYSTEM_LOG_MAX_PAGES

    controller = _Controller(
        legacy=_not_found(LEGACY),
        system_log_all=_paged([_threat(i) for i in range(IPS_SYSTEM_LOG_MAX_EVENTS + 500)]),
    )

    result, _ = await _call_tool(controller, limit=5000)

    assert len(controller.requests) == 1 + IPS_SYSTEM_LOG_MAX_PAGES
    assert result["count"] == IPS_SYSTEM_LOG_MAX_EVENTS
    assert result["max_events"] == IPS_SYSTEM_LOG_MAX_EVENTS
    assert result["truncated"] is True
    assert result["complete"] is False


# ---------------------------------------------------------------------------
# Legacy unavailable, system log unusable: never a successful zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system_log_answer",
    [
        _server_error(),
        _not_found(SYSTEM_LOG_ALL),
        # Shapes ConnectionManager.request can return: aiounifi wraps a JSON
        # object body as a one-element list, and a JSON array passes through.
        [{"unexpected": "envelope"}],
        [{}],
        [{"unexpected": "envelope", "data": "not-a-list"}],
        ["not json"],
        [{"data": [], "total_element_count": 1, "total_page_count": 1}],
        [{"data": []}],
        [],
        [{"data": [], "total_element_count": -1}],
        [{"data": [], "total_element_count": 0, "or_more": True}],
        [{"data": [], "total_element_count": 0, "total_page_count": 2, "or_more": True}],
    ],
    ids=[
        "http-500",
        "http-404",
        "object-without-event-list",
        "empty-object",
        "data-not-a-list",
        "string-row",
        "empty-page-with-total-1",
        "empty-page-without-total",
        "empty-array",
        "empty-page-with-negative-total",
        "lower-bound-zero",
        "lower-bound-zero-with-pages",
    ],
)
@pytest.mark.asyncio
async def test_no_usable_source_fails_instead_of_reporting_zero(system_log_answer):
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=system_log_answer)

    result, conn = await _call_tool(controller)

    assert result["success"] is False
    assert "count" not in result
    assert "/stat/ips/event" in result["error"]
    assert "/system-log/all" in result["error"]
    _assert_no_controller_address(result["error"])
    conn._update_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Legacy answers (behaviour on controllers that still serve it)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_events_are_returned_without_touching_the_system_log():
    legacy_events = [{"_id": "legacy-1", "src_ip": "198.51.100.40", "inner_alert_signature": "TEST SIGNATURE"}]
    controller = _Controller(legacy=legacy_events, system_log_all=_server_error())

    result, _ = await _call_tool(controller)

    assert result["success"] is True
    assert result["events"] == legacy_events
    assert result["count"] == 1
    assert result["source"] == "legacy"
    assert result["system_log_cross_check"] == "skipped"
    assert result["total"] is None
    assert result["complete"] is None
    assert controller.paths() == [LEGACY]


@pytest.mark.asyncio
async def test_empty_legacy_with_absent_system_log_is_an_uncached_legacy_zero():
    """(a) The system log is absent, corroborated by the count probe: the legacy answer stands."""
    controller = _Controller(
        legacy=[], system_log_all=_not_found(SYSTEM_LOG_ALL), system_log_count=_not_found(SYSTEM_LOG_COUNT)
    )

    result, conn = await _call_tool(controller)

    assert result["success"] is True
    assert result["count"] == 0
    assert result["source"] == "legacy"
    assert result["system_log_cross_check"] == "absent"
    assert "system_log_cross_check_error" not in result
    assert controller.paths() == [LEGACY, SYSTEM_LOG_ALL, SYSTEM_LOG_COUNT]
    # A 404 can also be a path or proxy problem, so the verdict is not held for the TTL.
    conn._update_cache.assert_not_called()


@pytest.mark.asyncio
async def test_404_from_system_log_all_with_a_live_count_probe_is_a_failure():
    """A 404 is not taken as absence when /system-log/count shows the system log exists."""
    controller = _Controller(legacy=[], system_log_all=_not_found(SYSTEM_LOG_ALL))

    result, conn = await _call_tool(controller)

    assert result["success"] is False
    assert result["system_log_cross_check"] == "failed"
    assert "/system-log/count answered" in result["system_log_cross_check_error"]
    conn._update_cache.assert_not_called()


@pytest.mark.parametrize(
    "system_log_error",
    [
        _server_error(),
        AiounifiException({"errorCode": 403, "message": "forbidden"}),
        ResponseError(
            f"Call https://{CONTROLLER_HOST}/proxy/network/v2/api/site/default/system-log/all received 429: b''"
        ),
        TimeoutError(),
        ResponseError(
            "Call https://192.0.2.44/proxy/network/v2/api/site/default/system-log/all received 500 Internal Server Error"
        ),
        AiounifiException({"errorCode": 500, "message": "request to https://192.0.2.44/proxy/network failed"}),
        RuntimeError("Cannot connect to host 192.0.2.1:443 ssl:default"),
        [{"unexpected": "envelope"}],
        [{"data": [], "total_element_count": 1, "total_page_count": 1}],
        [{"data": []}],
        [{"data": [], "total_element_count": 0, "or_more": True}],
        [{"data": [], "total_element_count": 0, "total_page_count": 2, "or_more": True}],
        AiounifiException({"errorCode": 500, "message": "Cannot connect to host unifi.example.:8443 ssl:default"}),
        RuntimeError("Cannot connect to host unifi:8443 ssl:default"),
    ],
    ids=[
        "http-500",
        "http-403",
        "http-429",
        "timeout",
        "aiounifi-500-with-url",
        "structured-message-with-url",
        "connect-error-with-host-port",
        "object-without-event-list",
        "empty-page-with-total-1",
        "empty-without-total",
        "lower-bound-zero",
        "lower-bound-zero-with-pages",
        "structured-message-with-fqdn-port",
        "connect-error-with-short-host",
    ],
)
@pytest.mark.asyncio
async def test_empty_legacy_with_failing_system_log_fails_closed(system_log_error):
    """(d) The system log exists but errors: no successful zero, diagnosis included, nothing cached."""
    controller = _Controller(legacy=[], system_log_all=system_log_error)

    result, conn = await _call_tool(controller)

    assert result["success"] is False
    assert "/stat/ips/event" in result["error"]
    assert "/system-log/all" in result["error"]
    assert result["source"] == "legacy"
    assert result["events"] == []
    assert result["window"]["duration_hours"] == 24
    assert result["system_log_cross_check"] == "failed"
    _assert_no_controller_address(result["error"])
    _assert_no_controller_address(result["system_log_cross_check_error"])
    conn._update_cache.assert_not_called()


@pytest.mark.asyncio
async def test_manager_list_keeps_returning_legacy_answer_when_cross_check_fails():
    """The REST/GraphQL list method keeps its behaviour for case (d)."""
    controller = _Controller(legacy=[], system_log_all=_server_error())
    manager = _manager(_connection(controller))

    assert await manager.get_ips_events() == []
    report = await manager.get_ips_event_report()
    assert report["system_log_cross_check"] == "failed"
    assert "internal error" in report["system_log_cross_check_error"]


@pytest.mark.asyncio
async def test_unreadable_legacy_shape_with_absent_system_log_fails():
    controller = _Controller(
        legacy={"unexpected": "envelope"},
        system_log_all=_not_found(SYSTEM_LOG_ALL),
        system_log_count=_not_found(SYSTEM_LOG_COUNT),
    )

    result, conn = await _call_tool(controller)

    assert result["success"] is False
    assert "unreadable shape" in result["error"]
    conn._update_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Failures after a page was read, and unreadable rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legacy", ["unavailable", "empty"])
@pytest.mark.parametrize(
    "later_error",
    [_not_found(SYSTEM_LOG_ALL), _server_error()],
    ids=["page-2-404", "page-2-500"],
)
@pytest.mark.asyncio
async def test_failure_after_a_page_was_read_is_not_a_zero(legacy, later_error):
    """Page one holds 100 threats of 101; page two fails. Never an empty legacy answer."""
    threats = [_threat(i, severity="HIGH") for i in range(101)]
    controller = _Controller(
        legacy=_not_found(LEGACY) if legacy == "unavailable" else [],
        system_log_all=_fail_on_page(threats, 1, later_error, total=101),
        system_log_count=_not_found(SYSTEM_LOG_COUNT),
    )

    result, conn = await _call_tool(controller, limit=200)

    assert result["success"] is False
    assert "page 2" in result["error"]
    assert "100 events were read" in result["error"]
    assert "count" not in result
    _assert_no_controller_address(result["error"])
    conn._update_cache.assert_not_called()
    # The count probe must not be consulted: a later-page failure proves the system log exists.
    assert SYSTEM_LOG_COUNT not in controller.paths()


@pytest.mark.parametrize(
    "rows",
    [[None], [_threat(i) for i in range(99)] + ["junk"], [_threat(1), 7]],
    ids=["null-row", "99-dicts-plus-junk", "number-row"],
)
@pytest.mark.parametrize("legacy", ["unavailable", "empty"])
@pytest.mark.asyncio
async def test_unreadable_rows_fail_instead_of_shrinking_the_page(rows, legacy):
    controller = _Controller(
        legacy=_not_found(LEGACY) if legacy == "unavailable" else [],
        system_log_all=_envelope(rows, len(rows)),
    )

    result, conn = await _call_tool(controller, limit=200)

    assert result["success"] is False
    assert "not an event object" in result["error"]
    conn._update_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Paging and coverage edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_pages_are_read_through_when_the_total_says_more_exists():
    """A server that serves 60-event pages: page one is short but the total is 120."""
    events = [_threat(i) for i in range(120)]

    def respond(data: dict) -> list:
        start = data["pageNumber"] * 60
        return _envelope(events[start : start + 60], 120)

    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=respond)

    result, _ = await _call_tool(controller, limit=500)

    assert result["count"] == 120
    assert result["complete"] is True
    assert [request.data["pageNumber"] for request in controller.requests[1:]] == [0, 1]


@pytest.mark.asyncio
async def test_lower_bound_flag_is_carried_to_pages_that_omit_it():
    events = [_threat(i) for i in range(100)]

    def respond(data: dict) -> list:
        if data["pageNumber"] == 0:
            return _envelope(events, 100, or_more=True)
        return [{"data": [], "page_number": data["pageNumber"]}]

    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=respond)

    result, _ = await _call_tool(controller, limit=500)

    assert result["count"] == 100
    assert result["complete"] is None
    assert result["truncated"] is None


@pytest.mark.asyncio
async def test_cap_reached_with_no_total_reports_truncation():
    from unifi_core.network.managers.stats_manager import IPS_SYSTEM_LOG_MAX_EVENTS, IPS_SYSTEM_LOG_MAX_PAGES

    def bare_pages(data: dict) -> list:
        return [_threat(data["pageNumber"] * 100 + i) for i in range(100)]

    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=bare_pages)

    result, _ = await _call_tool(controller, limit=5000)

    assert len(controller.requests) == 1 + IPS_SYSTEM_LOG_MAX_PAGES
    assert result["count"] == IPS_SYSTEM_LOG_MAX_EVENTS
    assert result["total"] is None
    assert result["truncated"] is True
    assert result["complete"] is False


@pytest.mark.asyncio
async def test_negative_total_is_unknown_not_complete():
    events = [_threat(i) for i in range(3)]
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=[{"data": events, "total_element_count": -1}])

    result, _ = await _call_tool(controller)

    assert result["success"] is True
    assert result["count"] == 3
    assert result["total"] is None
    assert result["complete"] is None


@pytest.mark.asyncio
async def test_logs_envelope_is_read_as_an_envelope():
    threat = _threat(1)
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=[{"logs": [threat], "total_element_count": 1}])

    result, _ = await _call_tool(controller)

    assert result["success"] is True
    assert result["events"] == [threat]
    assert result["complete"] is True


@pytest.mark.parametrize("legacy", ["unavailable", "empty"])
@pytest.mark.asyncio
async def test_empty_later_page_that_contradicts_the_total_fails(legacy):
    threats = [_threat(i) for i in range(100)]

    def respond(data: dict) -> list:
        if data["pageNumber"] == 0:
            return _envelope(threats, 150)
        return [{"data": [], "total_element_count": 150, "page_number": data["pageNumber"]}]

    controller = _Controller(legacy=_not_found(LEGACY) if legacy == "unavailable" else [], system_log_all=respond)

    result, conn = await _call_tool(controller, limit=500)

    assert result["success"] is False
    assert "page 2" in result["error"]
    assert "is empty" in result["error"]
    conn._update_cache.assert_not_called()
    assert len([r for r in controller.requests if r.path == SYSTEM_LOG_ALL]) == 2


@pytest.mark.parametrize("total", [150, 300])
@pytest.mark.asyncio
async def test_short_pages_are_counted_by_events_collected(total):
    """60-event server pages, exact total, no page count: every event is read."""
    events = [_threat(i) for i in range(total)]

    def respond(data: dict) -> list:
        start = data["pageNumber"] * 60
        return [{"data": events[start : start + 60], "total_element_count": total}]

    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=respond)

    result, _ = await _call_tool(controller, limit=500)

    assert result["count"] == total
    assert result["events"] == events
    assert result["complete"] is True
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_reached_exact_total_ends_the_read_despite_a_larger_page_count():
    events = [_threat(i) for i in range(100)]
    controller = _Controller(
        legacy=_not_found(LEGACY),
        system_log_all=[{"data": events, "total_element_count": 100, "total_page_count": 2}],
    )

    result, _ = await _call_tool(controller, limit=100)

    assert result["count"] == 100
    assert result["complete"] is True
    assert result["truncated"] is False
    assert len(controller.requests) == 2


@pytest.mark.parametrize(
    "probe_error",
    [AiounifiException({"errorCode": 403, "message": "forbidden"}), _server_error(), TimeoutError()],
    ids=["probe-403", "probe-500", "probe-timeout"],
)
@pytest.mark.asyncio
async def test_count_probe_failure_after_a_404_is_a_failed_cross_check(probe_error):
    controller = _Controller(legacy=[], system_log_all=_not_found(SYSTEM_LOG_ALL), system_log_count=probe_error)

    result, conn = await _call_tool(controller)

    assert result["success"] is False
    assert result["system_log_cross_check"] == "failed"
    assert "/system-log/count probe also failed" in result["system_log_cross_check_error"]
    _assert_no_controller_address(result["system_log_cross_check_error"])
    assert controller.paths() == [LEGACY, SYSTEM_LOG_ALL, SYSTEM_LOG_COUNT]
    conn._update_cache.assert_not_called()


@pytest.mark.asyncio
async def test_a_number_in_the_hostname_is_not_read_as_absence():
    """'status404' in a hostname must not turn two HTTP 500s into 'system log absent'."""
    error = ResponseError("Call https://status404.example/system-log/all received 500 Internal Server Error")
    probe_error = ResponseError("Call https://status404.example/system-log/count received 500 Internal Server Error")
    controller = _Controller(legacy=[], system_log_all=error, system_log_count=probe_error)

    result, conn = await _call_tool(controller)

    assert result["success"] is False
    assert result["system_log_cross_check"] == "failed"
    assert "HTTP 500" in result["system_log_cross_check_error"]
    assert "status404.example" not in result["system_log_cross_check_error"]
    conn._update_cache.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        ResponseError(
            "Call https://192.0.2.44/proxy/network/v2/api/site/default/system-log/all received 500 Internal Server Error"
        ),
        AiounifiException({"errorCode": 500, "message": "request to https://192.0.2.44/proxy/network failed"}),
        RuntimeError("Cannot connect to host 192.0.2.1:443 ssl:default"),
    ],
    ids=["aiounifi-500-with-url", "structured-message-with-url", "connect-error-with-host-port"],
)
@pytest.mark.asyncio
async def test_error_text_never_carries_a_controller_address(error):
    controller = _Controller(legacy=error, system_log_all=error)

    result, _ = await _call_tool(controller)

    assert result["success"] is False
    for address in ("192.0.2.44", "192.0.2.1", "://", ":443"):
        assert address not in result["error"]


@pytest.mark.parametrize("limit", [0, -5])
@pytest.mark.asyncio
async def test_non_positive_limit_is_rejected_without_a_request(limit):
    controller = _Controller(legacy=[], system_log_all=_envelope([], 0))

    result, conn = await _call_tool(controller, limit=limit)

    assert result["success"] is False
    assert "limit" in result["error"]
    assert controller.requests == []
    conn._update_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_returns_copies_of_the_report():
    threat = _threat(1, key="THREAT_BLOCKED_KNOWN_CLIENT")
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=_envelope([threat], 1))
    cache: dict = {}
    manager = _manager(_connection(controller, cache))

    first = await manager.get_ips_events()
    assert len(controller.requests) == 2
    first.clear()

    second = await manager.get_ips_events()
    assert isinstance(second, list)
    assert [event["key"] for event in second] == ["THREAT_BLOCKED_KNOWN_CLIENT"]
    second[0]["key"] = "MUTATED"

    report = await manager.get_ips_event_report()
    assert report["source"] == "system-log"
    assert report["events"][0]["key"] == "THREAT_BLOCKED_KNOWN_CLIENT"
    report["events"].append({"key": "ADDED"})
    report["source"] = "legacy"

    again = await manager.get_ips_event_report()
    assert again["source"] == "system-log"
    assert [event["key"] for event in again["events"]] == ["THREAT_BLOCKED_KNOWN_CLIENT"]
    assert len(controller.requests) == 2  # every later call was a cache hit


@pytest.mark.asyncio
async def test_empty_legacy_with_answering_system_log_uses_the_system_log():
    threat = _threat(1)
    controller = _Controller(legacy=[], system_log_all=_envelope([threat], 1))

    result, _ = await _call_tool(controller)

    assert result["source"] == "system-log"
    assert result["events"] == [threat]
    assert result["complete"] is True


@pytest.mark.asyncio
async def test_manager_list_api_still_returns_plain_events():
    """The REST/GraphQL surface reads get_ips_events and expects a list."""
    threat = _threat(1)
    controller = _Controller(legacy=_not_found(LEGACY), system_log_all=_envelope([threat], 1))
    manager = _manager(_connection(controller))

    assert await manager.get_ips_events() == [threat]


@pytest.mark.asyncio
async def test_call_prose_without_a_url_is_not_read_as_absence():
    """'Call parser received 404 bytes' is not an HTTP status, so the system log is not 'absent'."""
    error = RuntimeError("Call parser received 404 bytes")
    controller = _Controller(legacy=[], system_log_all=error, system_log_count=error)

    result, conn = await _call_tool(controller)

    assert result["success"] is False
    assert result["system_log_cross_check"] == "failed"
    conn._update_cache.assert_not_called()


@pytest.mark.parametrize("legacy", ["unavailable", "empty"])
@pytest.mark.asyncio
async def test_empty_page_under_a_lower_bound_total_with_pages_left_fails(legacy):
    threats = [_threat(i) for i in range(100)]

    def respond(data: dict) -> list:
        rows = threats if data["pageNumber"] == 0 else []
        return [{"data": rows, "total_element_count": 100, "total_page_count": 3, "or_more": True}]

    controller = _Controller(legacy=_not_found(LEGACY) if legacy == "unavailable" else [], system_log_all=respond)

    result, conn = await _call_tool(controller, limit=500)

    assert result["success"] is False
    assert "is empty" in result["error"]
    conn._update_cache.assert_not_called()
