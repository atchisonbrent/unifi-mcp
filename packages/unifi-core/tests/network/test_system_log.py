"""Tests for the shared v2 system-log helpers."""

import pytest
from aiounifi.errors import AiounifiException, ResponseError
from unifi_core.network.system_log import (
    EVENT_ROW_FIELDS,
    PageTracker,
    SystemLogResponseError,
    controller_message,
    coverage,
    describe_request_error,
    http_status,
    is_endpoint_absent,
    parse_system_log_page,
    safe_error_text,
    scrub_addresses,
)


def test_parse_list_wrapped_envelope():
    page = parse_system_log_page(
        [{"data": [{"id": "a"}], "total_element_count": 7, "total_page_count": 1, "or_more": False}]
    )
    assert page.events == [{"id": "a"}]
    assert (page.total, page.total_pages, page.total_is_lower_bound) == (7, 1, False)


def test_parse_reports_absent_or_more_as_unknown():
    assert parse_system_log_page([{"data": []}]).total_is_lower_bound is None


@pytest.mark.parametrize(
    "response",
    [[{"data": [{"id": "a"}, "junk"]}], [{"data": [None]}], [{"id": "a"}, 7]],
    ids=["junk-in-envelope", "null-row", "number-in-bare-list"],
)
def test_parse_rejects_non_object_rows(response):
    with pytest.raises(SystemLogResponseError, match="not an event object"):
        parse_system_log_page(response)


@pytest.mark.parametrize("bad", [-1, True, "7", 1.5])
def test_parse_treats_invalid_counts_as_absent(bad):
    page = parse_system_log_page([{"data": [], "total_element_count": bad, "total_page_count": bad}])
    assert page.total is None
    assert page.total_pages is None
    assert coverage(0, page.total)["complete"] is None


def test_parse_bare_list_has_unknown_total():
    page = parse_system_log_page([{"id": "a"}])
    assert page.events == [{"id": "a"}]
    assert page.total is None


@pytest.mark.parametrize("response", [{"unexpected": 1}, {"data": "x"}, None, "text", 3])
def test_parse_rejects_unreadable_shapes(response):
    with pytest.raises(SystemLogResponseError):
        parse_system_log_page(response)


@pytest.mark.parametrize(
    ("collected", "total", "lower_bound", "expected"),
    [
        (0, 0, False, (True, False)),
        (12, 12, False, (True, False)),
        (100, 101, False, (False, True)),
        (5, None, False, (None, None)),
        (12, 12, True, (None, None)),
        (11, 12, True, (False, True)),
    ],
)
def test_coverage_rule(collected, total, lower_bound, expected):
    result = coverage(collected, total, lower_bound)
    assert (result["complete"], result["truncated"]) == expected


@pytest.mark.parametrize("total", [None, 5, 1000])
def test_stopping_early_is_always_truncated(total):
    assert coverage(5, total, stopped_early=True) == {"complete": False, "truncated": True}


def _page(count, total=None, total_pages=None, or_more=None):
    return parse_system_log_page(
        [
            {
                "data": [{"id": str(i)} for i in range(count)],
                **({"total_element_count": total} if total is not None else {}),
                **({"total_page_count": total_pages} if total_pages is not None else {}),
                **({"or_more": or_more} if or_more is not None else {}),
            }
        ]
    )


def test_page_tracker_reads_past_a_short_page_when_counts_say_more():
    tracker = PageTracker(100)
    first = _page(60, total=120)
    tracker.observe(first)
    assert tracker.has_next_page(first, 0)


def test_page_tracker_stops_at_an_exact_total_and_on_empty_pages():
    tracker = PageTracker(100)
    page = _page(100, total=100)
    tracker.observe(page)
    assert not tracker.has_next_page(page, 0)
    assert not tracker.has_next_page(_page(0), 1)


def test_page_tracker_without_counts_uses_page_fullness():
    tracker = PageTracker(100)
    assert tracker.has_next_page(_page(100), 0)
    assert not tracker.has_next_page(_page(99), 0)


def test_page_tracker_carries_counts_and_lower_bound_forward():
    tracker = PageTracker(100)
    tracker.observe(_page(100, total=100, total_pages=1, or_more=True))
    later = _page(100)
    tracker.observe(later)
    assert (tracker.total, tracker.total_pages, tracker.total_is_lower_bound) == (100, 1, True)
    # A reached lower-bound total does not end the read while pages are full.
    assert tracker.has_next_page(later, 1)


def test_http_status_classification():
    not_found = ResponseError("Call https://192.0.2.1/x received 404 Not Found")
    server_error = AiounifiException({"errorCode": 500, "message": "boom"})
    assert http_status(not_found) == 404
    assert is_endpoint_absent(not_found)
    assert is_endpoint_absent(AiounifiException({"errorCode": 405, "message": "method"}))
    assert http_status(server_error) == 500
    assert not is_endpoint_absent(server_error)
    assert not is_endpoint_absent(TimeoutError())
    assert controller_message(server_error) == "boom"


def test_http_status_ignores_booleans():
    class _WithBoolStatus(Exception):
        status = True

    assert http_status(_WithBoolStatus("no code here")) is None
    assert http_status(AiounifiException({"errorCode": True, "message": "x"})) is None


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ResponseError("Call https://192.0.2.1/proxy/network/v2/api/site/default/x received 404 Not Found"),
            "HTTP 404 Not Found",
        ),
        (AiounifiException({"errorCode": 500, "message": "boom"}), "HTTP 500: boom"),
        (RuntimeError("Error requesting data from https://192.0.2.1:8443/api/x: refused"), None),
        (TimeoutError(), "TimeoutError"),
    ],
)
def test_describe_request_error_never_echoes_the_controller_address(error, expected):
    text = describe_request_error(error)
    assert "192.0.2.1" not in text
    assert "://" not in text
    if expected is not None:
        assert text == expected


# ---------------------------------------------------------------------------
# Second review round
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [[{"logs": [{"id": "a"}], "total_element_count": 1}], [{"data": [{"id": "a"}], "total_element_count": 1}]],
    ids=["logs", "data"],
)
def test_parse_reads_a_one_element_list_holding_an_envelope(response):
    page = parse_system_log_page(response)
    assert page.events == [{"id": "a"}]
    assert page.total == 1


@pytest.mark.parametrize(
    "response",
    [[{}], [{"unexpected": 1}], [{"id": "a"}, {"msg": "no identity"}], ["not json"]],
    ids=["empty-object", "object-without-event-list", "row-without-identity", "string-row"],
)
def test_parse_requires_bare_list_rows_to_look_like_events(response):
    with pytest.raises(SystemLogResponseError):
        parse_system_log_page(response)


@pytest.mark.parametrize("field", sorted(EVENT_ROW_FIELDS))
def test_any_identifying_field_makes_a_bare_row_an_event(field):
    assert parse_system_log_page([{field: "x"}]).events == [{field: "x"}]


def test_page_tracker_counts_events_actually_collected():
    tracker = PageTracker(100)
    for page_number in range(4):
        page = _page(60, total=300)
        tracker.observe(page)
        assert tracker.has_next_page(page, page_number), page_number
    assert tracker.served == 240


def test_page_tracker_base_offsets_a_later_start():
    tracker = PageTracker(100, base=200)
    page = _page(50, total=250)
    tracker.observe(page)
    assert tracker.served == 250
    assert not tracker.has_next_page(page, 2)


def test_page_tracker_exact_total_wins_over_page_count():
    tracker = PageTracker(100)
    page = _page(100, total=100, total_pages=2)
    tracker.observe(page)
    assert not tracker.has_next_page(page, 0)


@pytest.mark.parametrize(
    ("total", "total_pages"),
    [(1, None), (1, 1), (None, 3), (150, None)],
    ids=["total-1", "total-1-pages-1", "pages-3-no-total", "total-150"],
)
def test_page_tracker_rejects_an_empty_page_that_contradicts_its_counts(total, total_pages):
    tracker = PageTracker(100)
    page = _page(0, total=total, total_pages=total_pages)
    tracker.observe(page)
    with pytest.raises(SystemLogResponseError, match="is empty"):
        tracker.has_next_page(page, 0)


def test_page_tracker_accepts_an_empty_page_once_the_total_is_reached():
    tracker = PageTracker(100)
    tracker.observe(_page(100, total=100, total_pages=2))
    empty = _page(0)
    tracker.observe(empty)
    assert not tracker.has_next_page(empty, 1)


@pytest.mark.parametrize(
    "error",
    [
        ResponseError("Call https://status404.example/system-log/all received 500 Internal Server Error"),
        RuntimeError("payload received 404 bytes"),
        RuntimeError("GET https://192.0.2.1/x?status=404 failed"),
        RuntimeError("status: 404"),
        AiounifiException({"errorCode": 4040, "message": "not an HTTP code"}),
        RuntimeError("Call parser received 404 bytes"),
        RuntimeError("Call ftp://192.0.2.1/x received 404 Not Found"),
    ],
    ids=[
        "hostname-status404",
        "byte-count",
        "query-string",
        "free-text",
        "out-of-range-code",
        "call-without-url",
        "call-with-non-http-url",
    ],
)
def test_http_status_does_not_read_arbitrary_numbers_as_404(error):
    assert http_status(error) != 404
    assert not is_endpoint_absent(error)


def test_http_status_reads_the_aiounifi_form_and_structured_codes():
    assert http_status(ResponseError("Call https://192.0.2.1/x received 404 Not Found")) == 404
    assert http_status(ResponseError("Call https://192.0.2.1/x received 429: b'busy'")) == 429
    assert http_status(AiounifiException({"errorCode": 405, "message": "x"})) == 405

    class _WithStatus(Exception):
        status = 503

    assert http_status(_WithStatus("Call https://status404.example/x received 404 Not Found")) == 503


@pytest.mark.parametrize(
    "text",
    [
        "see https://192.0.2.44/proxy/network",
        "host 192.0.2.1:443 refused",
        "host 192.0.2.1 refused",
        "host [2001:db8::1]:8443 refused",
        "host 2001:db8::2 refused",
        "host unifi.example:8443 refused",
    ],
)
def test_scrub_addresses_removes_urls_addresses_and_host_ports(text):
    scrubbed = scrub_addresses(text)
    for fragment in ("192.0.2", "2001:db8", "unifi.example", "://", ":443", ":8443"):
        assert fragment not in scrubbed


@pytest.mark.parametrize(
    "error",
    [
        AiounifiException({"errorCode": 500, "message": "request to https://192.0.2.44/proxy/network failed"}),
        AiounifiException({"errorCode": 400, "message": "bad filter from 192.0.2.44:8443"}),
        RuntimeError("Cannot connect to host 192.0.2.1:443 ssl:default"),
        ResponseError("Call https://192.0.2.44/proxy/network/x received 500 Internal Server Error"),
    ],
)
def test_describe_and_controller_message_never_echo_an_address(error):
    for text in (describe_request_error(error), controller_message(error), safe_error_text(error)):
        assert "192.0.2" not in text
        assert "://" not in text


def test_safe_error_text_scrubs_value_errors_too():
    text = safe_error_text(ValueError("rejected input https://192.0.2.44/x from 192.0.2.1:8443"))
    assert "192.0.2" not in text
    assert "://" not in text
    assert text.startswith("rejected input")


def test_safe_error_text_scrubs_pydantic_validation_errors():
    from pydantic import BaseModel, ValidationError

    class _Model(BaseModel):
        id: str

    with pytest.raises(ValidationError) as caught:
        _Model(id={"href": "https://192.0.2.44/x"})
    assert "192.0.2.44" in str(caught.value)
    text = safe_error_text(caught.value)
    assert "192.0.2.44" not in text
    assert "validation error" in text


def test_page_tracker_lower_bound_total_does_not_override_the_page_count():
    tracker = PageTracker(100)
    first = _page(100, total=100, total_pages=3, or_more=True)
    tracker.observe(first)
    assert tracker.has_next_page(first, 0)
    empty = _page(0, total=100, total_pages=3, or_more=True)
    tracker.observe(empty)
    with pytest.raises(SystemLogResponseError, match="is empty"):
        tracker.has_next_page(empty, 1)


@pytest.mark.parametrize(
    "text",
    [
        "Cannot connect to host unifi.example.:8443 ssl:default",
        "Cannot connect to host unifi:8443 ssl:default",
        "Cannot connect to host localhost:8443 ssl:default",
        "upstream unifi.example:8443 timed out",
        "upstream localhost:8443 timed out",
    ],
)
def test_scrub_addresses_removes_hosts(text):
    scrubbed = scrub_addresses(text)
    assert "unifi" not in scrubbed
    assert "localhost" not in scrubbed
    assert ":8443" not in scrubbed


@pytest.mark.parametrize(
    "text",
    [
        "Invalid filter keys:123",
        "BOGUS:1234 is not a category",
        "client 02:00:00:00:00:01 disconnected",
        "window 09:30:00 to 10:00:00",
        "retry with host header set",
        "ssl:default",
    ],
)
def test_scrub_addresses_keeps_identifiers_that_only_look_like_addresses(text):
    assert scrub_addresses(text) == text
