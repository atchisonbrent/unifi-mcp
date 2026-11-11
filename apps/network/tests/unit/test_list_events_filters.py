"""``unifi_list_events`` filters and coverage, through the tool and the real EventManager.

The controller is mocked at ``ConnectionManager.request``. Fixtures are
synthetic (documentation address ranges, invented names).
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiounifi.errors import AiounifiException, ResponseError

os.environ.setdefault("UNIFI_HOST", "127.0.0.1")
os.environ.setdefault("UNIFI_USERNAME", "test")
os.environ.setdefault("UNIFI_PASSWORD", "test")


def _event(index: int, category: str = "CLIENT_DEVICES", subcategory: str = "WIFI") -> dict:
    return {
        "id": f"evt-{index:04d}",
        "key": "CLIENT_CONNECTED_WIRELESS_2",
        "category": category,
        "subcategory": subcategory,
        "severity": "LOW",
        "timestamp": 1786000000000 - index * 1000,
        "message_raw": "{CLIENT} connected.",
        "parameters": {"CLIENT": {"id": "02:00:00:00:00:02", "ip": "192.0.2.30", "name": "Test-Phone"}},
    }


def _envelope(events: list, total: int | None, page_size: int = 100) -> list:
    envelope = {"data": events, "page_number": 0}
    if total is not None:
        envelope["total_element_count"] = total
        envelope["total_page_count"] = -(-total // page_size) if total else 0
    return [envelope]


def _connection(response=None, side_effect=None) -> MagicMock:
    conn = MagicMock()
    conn.site = "default"
    conn.request = AsyncMock(return_value=response, side_effect=side_effect)
    return conn


@pytest.fixture
def tool(monkeypatch):
    """Return (list_events, install) where install(conn, use_v2) wires a real EventManager."""
    from unifi_core.network.managers.event_manager import EventManager
    from unifi_network_mcp.tools import events as events_module

    def install(conn, use_v2=True):
        manager = EventManager(conn)
        manager._use_v2 = use_v2
        monkeypatch.setattr(events_module, "_get_event_manager", lambda: manager)
        return manager

    return events_module.list_events, install


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_above_returned_sets_truncated(tool):
    list_events, install = tool
    install(_connection(_envelope([_event(i) for i in range(10)], 250, page_size=10)))

    result = await list_events(limit=10)

    assert result["success"] is True
    assert result["count"] == 10
    assert result["total"] == 250
    assert result["truncated"] is True
    assert result["complete"] is False
    assert result["has_more"] is True
    assert result["source"] == "system-log"


@pytest.mark.asyncio
async def test_total_equal_to_returned_is_complete(tool):
    list_events, install = tool
    install(_connection(_envelope([_event(i) for i in range(12)], 12)))

    result = await list_events()

    assert (result["count"], result["total"]) == (12, 12)
    assert result["complete"] is True
    assert result["truncated"] is False
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_missing_total_leaves_coverage_unknown(tool):
    list_events, install = tool
    install(_connection([_event(1)]))

    result = await list_events()

    assert result["total"] is None
    assert result["complete"] is None
    assert result["truncated"] is None


@pytest.mark.asyncio
async def test_a_later_page_never_claims_the_whole_window(tool):
    """start skips events, so even the final page is not complete coverage."""
    list_events, install = tool
    install(_connection(_envelope([_event(i) for i in range(10, 15)], 15, page_size=10)))

    result = await list_events(limit=10, start=10)

    assert result["count"] == 5
    assert result["truncated"] is True
    assert result["complete"] is False
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_a_later_page_of_an_empty_window_is_not_complete(tool):
    list_events, install = tool
    install(_connection(_envelope([], 0)))

    result = await list_events(start=10)

    assert result["count"] == 0
    assert result["complete"] is False


@pytest.mark.parametrize(
    ("total", "or_more", "has_more"),
    [(250, False, True), (250, True, True), (10, False, False), (10, True, None), (None, False, None)],
    ids=["below-exact", "below-lower-bound", "exact-reached", "lower-bound-reached", "no-total"],
)
@pytest.mark.asyncio
async def test_has_more_is_tri_state(tool, total, or_more, has_more):
    list_events, install = tool
    envelope = {"data": [_event(i) for i in range(10)], "page_number": 0}
    if total is not None:
        envelope["total_element_count"] = total
        envelope["or_more"] = or_more
    install(_connection([envelope]))

    result = await list_events(limit=10)

    assert result["has_more"] is has_more
    if or_more and total == 10:
        # The read stopped at the limit with a full page under a lower-bound total:
        # never reported complete.
        assert result["complete"] is False
        assert result["truncated"] is True


@pytest.mark.asyncio
async def test_short_first_page_is_read_through_when_the_total_says_more(tool):
    """A server that serves 60-event pages for a 100-event request."""
    list_events, install = tool
    events = [_event(i) for i in range(120)]

    def respond(api_request):
        start = api_request.data["pageNumber"] * 60
        return _envelope(events[start : start + 60], 120)

    conn = _connection(side_effect=respond)
    install(conn)

    result = await list_events(limit=100)

    assert result["count"] == 100
    assert result["total"] == 120
    assert result["truncated"] is True
    assert result["has_more"] is True
    assert conn.request.await_count == 2


@pytest.mark.asyncio
async def test_failure_after_a_page_was_read_fails_the_call(tool):
    list_events, install = tool
    first = _envelope([_event(i) for i in range(100)], 150)
    conn = _connection(side_effect=[first, AiounifiException({"errorCode": 500, "message": "internal error"})])
    install(conn)

    result = await list_events(limit=150)

    assert result["success"] is False
    assert "page 2 failed after 100 events were read" in result["error"]


@pytest.mark.asyncio
async def test_unreadable_rows_fail_the_call(tool):
    list_events, install = tool
    install(_connection(_envelope([_event(1), None], 2)))

    result = await list_events()

    assert result["success"] is False
    assert "not an event object" in result["error"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": -1}, "start must be an integer of at least 0"),
        ({"within_hours": 0}, "within_hours must be an integer of at least 1"),
        ({"within_hours": -3}, "within_hours must be an integer of at least 1"),
        ({"limit": 0}, "limit must be an integer of at least 1"),
        ({"event_type": 42}, "event_type must be a string"),
        ({"event_type": ["CLIENT_CONNECTED_WIRELESS_2"]}, "event_type must be a string"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_query_arguments_are_rejected_without_a_request(tool, kwargs, message):
    list_events, install = tool
    conn = _connection(_envelope([], 0))
    install(conn, use_v2=None)

    result = await list_events(**kwargs)

    assert result["success"] is False
    assert message in result["error"]
    conn.request.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        # Shapes ConnectionManager.request can return: aiounifi wraps a JSON
        # object body as a one-element list, and a JSON array passes through.
        [{"unexpected": "envelope"}],
        [{}],
        [{"data": "not-a-list"}],
        ["not json"],
        [{"data": [], "total_element_count": 1, "total_page_count": 1}],
    ],
    ids=["object-without-event-list", "empty-object", "data-not-a-list", "string-row", "empty-page-with-total-1"],
)
@pytest.mark.asyncio
async def test_unreadable_response_fails_instead_of_returning_nothing(tool, response):
    list_events, install = tool
    install(_connection(response))

    result = await list_events()

    assert result["success"] is False
    assert "Failed to list events" in result["error"]


@pytest.mark.asyncio
async def test_logs_envelope_is_read_as_an_envelope(tool):
    list_events, install = tool
    install(_connection([{"logs": [_event(1)], "total_element_count": 1}]))

    result = await list_events()

    assert result["success"] is True
    assert result["count"] == 1
    assert result["complete"] is True


@pytest.mark.asyncio
async def test_empty_later_page_that_contradicts_the_total_fails(tool):
    list_events, install = tool
    first = [{"data": [_event(i) for i in range(100)], "total_element_count": 150}]
    empty = [{"data": [], "total_element_count": 150}]
    conn = _connection(side_effect=[first, empty, empty])
    install(conn)

    result = await list_events(limit=500)

    assert result["success"] is False
    assert "page 2" in result["error"]
    assert "is empty" in result["error"]
    assert conn.request.await_count == 2


@pytest.mark.parametrize("total", [150, 300])
@pytest.mark.asyncio
async def test_short_pages_are_counted_by_events_collected(tool, total):
    """60-event server pages, exact total, no page count: every event is read."""
    list_events, install = tool
    events = [_event(i) for i in range(total)]

    def respond(api_request):
        start = api_request.data["pageNumber"] * 60
        return [{"data": events[start : start + 60], "total_element_count": total}]

    install(_connection(side_effect=respond))

    result = await list_events(limit=500)

    assert result["count"] == total
    assert [event["id"] for event in result["events"]] == [event["id"] for event in events]
    assert result["complete"] is True
    assert result["truncated"] is False
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_reached_exact_total_ends_the_read_despite_a_larger_page_count(tool):
    list_events, install = tool
    conn = _connection([{"data": [_event(i) for i in range(100)], "total_element_count": 100, "total_page_count": 2}])
    install(conn)

    result = await list_events(limit=100)

    assert result["count"] == 100
    assert result["complete"] is True
    assert result["truncated"] is False
    assert result["has_more"] is False
    assert conn.request.await_count == 1


@pytest.mark.asyncio
async def test_page_cap_bounds_a_pager_that_never_runs_out(tool):
    from unifi_core.network.managers.event_manager import EVENT_LIST_MAX_PAGES

    list_events, install = tool

    def endless(api_request):
        base = api_request.data["pageNumber"] * 100
        return [{"data": [_event(base + i) for i in range(100)]}]

    conn = _connection(side_effect=endless)
    install(conn)

    result = await list_events(limit=1_000_000)

    assert conn.request.await_count == EVENT_LIST_MAX_PAGES
    assert result["count"] == EVENT_LIST_MAX_PAGES * 100
    assert result["truncated"] is True
    assert result["complete"] is False


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
@pytest.mark.parametrize("path", ["system-log", "legacy-after-failed-probe"])
@pytest.mark.asyncio
async def test_error_text_never_carries_a_controller_address(tool, error, path):
    list_events, install = tool
    if path == "system-log":
        conn = _connection(side_effect=error)
        install(conn)
    else:
        # The version probe fails with the error, then legacy /stat/event fails too.
        conn = _connection(side_effect=[error, error])
        install(conn, use_v2=None)

    result = await list_events()

    assert result["success"] is False
    for address in ("192.0.2.44", "192.0.2.1", "://", ":443"):
        assert address not in result["error"]
    if path == "legacy-after-failed-probe":
        assert "v2 system-log probe failed" in result["error"]


@pytest.mark.asyncio
async def test_a_controller_400_message_is_scrubbed(tool):
    list_events, install = tool
    rejection = AiounifiException(
        {"errorCode": 400, "message": "Unknown category NEW_CATEGORY at https://192.0.2.44/x"}
    )
    install(_connection(side_effect=rejection))

    result = await list_events(categories=["NEW_CATEGORY"])

    assert result["success"] is False
    assert "Unknown category NEW_CATEGORY" in result["error"]
    assert "192.0.2.44" not in result["error"]


# ---------------------------------------------------------------------------
# Event fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_carry_category_and_subcategory(tool):
    list_events, install = tool
    install(_connection(_envelope([_event(1, category="SECURITY", subcategory="INTRUSION_PREVENTION")], 1)))

    event = (await list_events())["events"][0]

    assert event["category"] == "SECURITY"
    assert event["subcategory"] == "INTRUSION_PREVENTION"


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_severities_and_categories_are_sent_and_echoed(tool):
    list_events, install = tool
    conn = _connection(_envelope([], 0))
    install(conn)

    result = await list_events(categories=["SECURITY", "VPN"], severities=["HIGH", "VERY_HIGH"])

    body = conn.request.call_args.args[0].data
    assert body["severities"] == ["HIGH", "VERY_HIGH"]
    assert body["categories"] == ["SECURITY", "VPN"]
    assert result["filters"]["severities"] == ["HIGH", "VERY_HIGH"]
    assert result["filters"]["categories"] == ["SECURITY", "VPN"]


@pytest.mark.asyncio
async def test_default_filters_are_echoed_as_applied(tool):
    from unifi_core.network.managers.event_manager import KNOWN_EVENT_CATEGORIES

    list_events, install = tool
    conn = _connection(_envelope([], 0))
    install(conn)

    result = await list_events()

    assert result["filters"]["categories"] == list(KNOWN_EVENT_CATEGORIES)
    assert result["filters"]["severities"] == ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    assert conn.request.call_args.args[0].data["categories"] == list(KNOWN_EVENT_CATEGORIES)


@pytest.mark.parametrize(
    "categories",
    [["security"], ["EVT_IPS_"], [""], ["SECURITY", 7], ["SECURITY", "has space"], ["SECURITY\n"]],
    ids=["lowercase", "evt-prefix", "empty", "non-string", "space", "trailing-newline"],
)
@pytest.mark.asyncio
async def test_malformed_category_is_rejected_without_a_request(tool, categories):
    list_events, install = tool
    conn = _connection(_envelope([], 0))
    install(conn, use_v2=None)

    result = await list_events(categories=categories)

    assert result["success"] is False
    assert "Invalid category" in result["error"]
    assert "SECURITY" in result["error"]
    conn.request.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_severity_is_rejected_without_a_request(tool):
    list_events, install = tool
    conn = _connection(_envelope([], 0))
    install(conn, use_v2=None)

    result = await list_events(severities=["CRITICAL"])

    assert result["success"] is False
    assert "VERY_HIGH" in result["error"]
    conn.request.assert_not_called()


@pytest.mark.asyncio
async def test_well_formed_category_outside_the_known_list_is_sent(tool):
    list_events, install = tool
    conn = _connection(_envelope([_event(1, category="NEW_CATEGORY")], 1))
    install(conn)

    result = await list_events(categories=["NEW_CATEGORY"])

    assert result["success"] is True
    assert conn.request.call_args.args[0].data["categories"] == ["NEW_CATEGORY"]


@pytest.mark.asyncio
async def test_controller_rejection_of_a_category_carries_its_message_and_the_known_list(tool):
    list_events, install = tool
    rejection = AiounifiException({"errorCode": 400, "message": "Unknown category NEW_CATEGORY"})
    install(_connection(side_effect=rejection))

    result = await list_events(categories=["NEW_CATEGORY"])

    assert result["success"] is False
    assert "Unknown category NEW_CATEGORY" in result["error"]
    assert "CLIENT_DEVICES" in result["error"]


@pytest.mark.parametrize("event_type", ["EVT_GW_", "EVT_WU_Connected", "evt_ips_"])
@pytest.mark.asyncio
async def test_legacy_prefix_is_rejected_with_a_pointer_and_no_request(tool, event_type):
    list_events, install = tool
    conn = _connection(_envelope([], 0))
    install(conn)

    result = await list_events(event_type=event_type)

    assert result["success"] is False
    assert "EVT_*" in result["error"]
    assert "categories" in result["error"]
    assert "exact event key" in result["error"]
    conn.request.assert_not_called()


@pytest.mark.asyncio
async def test_exact_key_passes_through(tool):
    list_events, install = tool
    conn = _connection(_envelope([_event(1)], 1))
    install(conn)

    result = await list_events(event_type="CLIENT_CONNECTED_WIRELESS_2")

    assert result["success"] is True
    body = conn.request.call_args.args[0].data
    assert body["keys"] == ["CLIENT_CONNECTED_WIRELESS_2"]
    assert result["filters"]["event_type"] == "CLIENT_CONNECTED_WIRELESS_2"


@pytest.mark.parametrize(
    "filters",
    [{"categories": ["SECURITY"]}, {"severities": ["HIGH"]}],
    ids=["categories", "severities"],
)
@pytest.mark.asyncio
async def test_system_log_filters_on_a_legacy_controller_are_an_error(tool, filters):
    list_events, install = tool
    conn = _connection([{"_id": "legacy-1", "key": "EVT_WU_Connected"}])
    install(conn, use_v2=False)

    result = await list_events(**filters)

    assert result["success"] is False
    assert "require the system log" in result["error"]
    assert "legacy events API" in result["error"]
    conn.request.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_controller_still_accepts_evt_types(tool):
    """EVT_* keys are native to /stat/event; only the system log rejects them."""
    list_events, install = tool
    conn = _connection([{"_id": "legacy-1", "key": "EVT_WU_Connected", "user": "02:00:00:00:00:03"}])
    install(conn, use_v2=False)

    result = await list_events(event_type="EVT_WU_Connected")

    assert result["success"] is True
    assert result["source"] == "legacy"
    assert result["total"] is None
    assert result["filters"]["categories"] is None
    request = conn.request.call_args.args[0]
    assert request.path == "/stat/event"
    assert request.data["type"] == "EVT_WU_Connected"


# ---------------------------------------------------------------------------
# Offsets, lower-bound totals and error text
# ---------------------------------------------------------------------------


def _numbered_by(stride: int, events: list, total: int | None):
    """A server that serves ``stride`` events per page and numbers its pages by ``stride``.

    With ``total`` None the envelope carries no counts at all.
    """

    def respond(api_request):
        start = api_request.data["pageNumber"] * stride
        envelope = {"data": events[start : start + stride]}
        if total is not None:
            envelope["total_element_count"] = total
        return [envelope]

    return respond


def _numbered_by_60(events: list, total: int | None):
    """A server that serves 60 events per page and numbers its pages by 60."""
    return _numbered_by(60, events, total)


@pytest.mark.parametrize(
    ("start", "expected_ids"),
    [(100, range(100, 200)), (160, range(160, 200))],
    ids=["start-100", "start-160"],
)
@pytest.mark.asyncio
async def test_later_start_on_a_short_page_server_returns_the_requested_window(tool, start, expected_ids):
    list_events, install = tool
    events = [_event(i) for i in range(200)]
    install(_connection(side_effect=_numbered_by_60(events, 200)))

    result = await list_events(start=start, limit=100)

    assert result["success"] is True
    assert [event["id"] for event in result["events"]] == [f"evt-{i:04d}" for i in expected_ids]
    assert result["has_more"] is False
    assert result["complete"] is False


@pytest.mark.asyncio
async def test_later_start_on_a_normal_server_still_jumps_straight_to_the_page(tool):
    list_events, install = tool
    events = [_event(i) for i in range(400)]

    def respond(api_request):
        start = api_request.data["pageNumber"] * 100
        return [{"data": events[start : start + 100], "total_element_count": 400}]

    conn = _connection(side_effect=respond)
    install(conn)

    result = await list_events(start=200, limit=100)

    assert conn.request.await_count == 1
    assert conn.request.call_args.args[0].data["pageNumber"] == 2
    assert [event["id"] for event in result["events"]] == [f"evt-{i:04d}" for i in range(200, 300)]
    assert result["has_more"] is True


@pytest.mark.asyncio
async def test_realigned_read_keeps_the_page_cap(tool):
    from unifi_core.network.managers.event_manager import EVENT_LIST_MAX_PAGES

    list_events, install = tool
    total = 60 * (EVENT_LIST_MAX_PAGES + 20)
    events = [_event(i) for i in range(total)]
    conn = _connection(side_effect=_numbered_by_60(events, total))
    install(conn)

    result = await list_events(start=60 * (EVENT_LIST_MAX_PAGES + 5), limit=100)

    assert conn.request.await_count == EVENT_LIST_MAX_PAGES
    assert result["count"] == 0
    assert result["truncated"] is True
    assert result["has_more"] is True


@pytest.mark.parametrize(
    ("total", "count", "expected_ids", "has_more"),
    [(160, 160, range(100, 160), False), (None, 200, range(100, 200), None)],
    ids=["exact-total-fits-the-short-page", "no-counts"],
)
@pytest.mark.asyncio
async def test_short_jumped_page_that_counts_accept_is_verified_against_page_zero(
    tool, total, count, expected_ids, has_more
):
    """Counts cannot tell a short final page from a server that pages by 60; page 0 can."""
    list_events, install = tool
    events = [_event(i) for i in range(count)]
    conn = _connection(side_effect=_numbered_by_60(events, total))
    install(conn)

    result = await list_events(start=100, limit=100)

    assert result["success"] is True
    assert [event["id"] for event in result["events"]] == [f"evt-{i:04d}" for i in expected_ids]
    assert result["has_more"] is has_more
    requested = [call.args[0].data["pageNumber"] for call in conn.request.call_args_list]
    assert requested[:2] == [1, 0]
    assert requested.count(0) == 1, "the page 0 fetched by the check is reused, not requested again"


@pytest.mark.asyncio
async def test_short_final_page_on_a_normal_server_costs_one_check(tool):
    list_events, install = tool
    events = [_event(i) for i in range(250)]
    conn = _connection(side_effect=_numbered_by(100, events, 250))
    install(conn)

    result = await list_events(start=200, limit=100)

    assert [event["id"] for event in result["events"]] == [f"evt-{i:04d}" for i in range(200, 250)]
    assert result["has_more"] is False
    assert [call.args[0].data["pageNumber"] for call in conn.request.call_args_list] == [2, 0]


@pytest.mark.asyncio
async def test_full_jumped_page_makes_no_extra_request(tool):
    list_events, install = tool
    events = [_event(i) for i in range(300)]
    conn = _connection(side_effect=_numbered_by(100, events, 300))
    install(conn)

    result = await list_events(start=100, limit=100)

    assert [event["id"] for event in result["events"]] == [f"evt-{i:04d}" for i in range(100, 200)]
    assert result["has_more"] is True
    assert conn.request.await_count == 1


@pytest.mark.asyncio
async def test_start_beyond_the_total_makes_one_request(tool):
    list_events, install = tool
    events = [_event(i) for i in range(150)]
    conn = _connection(side_effect=_numbered_by(100, events, 150))
    install(conn)

    result = await list_events(start=300, limit=100)

    assert result["success"] is True
    assert result["count"] == 0
    assert conn.request.await_count == 1


@pytest.mark.asyncio
async def test_failed_page_zero_check_fails_the_read(tool):
    list_events, install = tool
    jumped = [{"data": [_event(i) for i in range(200, 250)], "total_element_count": 250}]
    install(_connection(side_effect=[jumped, RuntimeError("Cannot connect to host unifi.example:8443")]))

    result = await list_events(start=200, limit=100)

    assert result["success"] is False
    assert "page 1 failed" in result["error"]
    assert "unifi.example" not in result["error"]


@pytest.mark.asyncio
async def test_empty_page_zero_after_a_non_empty_jumped_page_fails(tool):
    list_events, install = tool
    jumped = [{"data": [_event(i) for i in range(200, 250)]}]
    install(_connection(side_effect=[jumped, [{"data": []}]]))

    result = await list_events(start=200, limit=100)

    assert result["success"] is False
    assert "page 1 is empty" in result["error"]


@pytest.mark.asyncio
async def test_empty_page_zero_fails_even_when_the_counts_flagged_the_jumped_page(tool):
    """The count check must not skip the empty-page-0 rejection."""
    list_events, install = tool
    jumped = [{"data": [_event(i) for i in range(60, 120)], "total_element_count": 200}]
    install(_connection(side_effect=[jumped, [{"data": []}]]))

    result = await list_events(start=100, limit=100)

    assert result["success"] is False
    assert "page 1 is empty" in result["error"]


@pytest.mark.asyncio
async def test_full_page_zero_keeps_the_jump_even_when_the_counts_flagged_it(tool):
    """A server whose pages vary in length (100, 60, 40) still numbers them by the requested size."""
    list_events, install = tool
    events = [_event(i) for i in range(200)]
    pages = {0: events[0:100], 1: events[100:160], 2: events[160:200]}
    requested = []

    def respond(api_request):
        requested.append(api_request.data["pageNumber"])
        return [{"data": pages.get(api_request.data["pageNumber"], []), "total_element_count": 200}]

    install(_connection(side_effect=respond))

    result = await list_events(start=100, limit=100)

    assert result["success"] is True
    assert [event["id"] for event in result["events"]] == [f"evt-{i:04d}" for i in range(100, 200)]
    assert requested == [1, 0, 2]
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_empty_page_under_a_lower_bound_total_with_pages_left_fails(tool):
    list_events, install = tool
    first = [
        {"data": [_event(i) for i in range(100)], "total_element_count": 100, "total_page_count": 3, "or_more": True}
    ]
    empty = [{"data": [], "total_element_count": 100, "total_page_count": 3, "or_more": True}]
    install(_connection(side_effect=[first, empty]))

    result = await list_events(limit=500)

    assert result["success"] is False
    assert "is empty" in result["error"]


@pytest.mark.asyncio
async def test_validation_error_from_controller_data_is_scrubbed(tool):
    """A pydantic ValidationError is a ValueError and echoes the controller-supplied value."""
    list_events, install = tool
    bad = {"id": {"href": "https://192.0.2.44/x"}, "key": "CLIENT_CONNECTED_WIRELESS_2", "timestamp": 1786000000000}
    install(_connection([{"data": [bad], "total_element_count": 1}]))

    result = await list_events()

    assert result["success"] is False
    assert "192.0.2.44" not in result["error"]
    assert "://" not in result["error"]
    assert "validation error" in result["error"]


@pytest.mark.asyncio
async def test_controller_400_message_still_names_the_rejected_filter(tool):
    list_events, install = tool
    rejection = AiounifiException({"errorCode": 400, "message": "Invalid filter keys:123"})
    install(_connection(side_effect=rejection))

    result = await list_events(event_type="123")

    assert result["success"] is False
    assert "Invalid filter keys:123" in result["error"]


@pytest.mark.parametrize(
    "error",
    [
        AiounifiException({"errorCode": 500, "message": "Cannot connect to host unifi.example.:8443 ssl:default"}),
        RuntimeError("Cannot connect to host unifi:8443 ssl:default"),
    ],
    ids=["structured-fqdn-with-trailing-dot", "connect-error-short-host"],
)
@pytest.mark.asyncio
async def test_host_forms_are_scrubbed_from_the_tool_error(tool, error):
    list_events, install = tool
    install(_connection(side_effect=error))

    result = await list_events()

    assert result["success"] is False
    assert "unifi" not in result["error"]
    assert ":8443" not in result["error"]
