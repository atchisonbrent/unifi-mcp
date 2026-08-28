"""MAC lists must reach the controller lowercased, without losing their siblings.

The MCP tool layer and the apps/api dispatch each assemble these payloads
themselves, so neither a model validator nor a ``to_controller_*`` builder is on
the path — the manager is the one boundary they share, and normalising there is
what this change does.

Every test here asserts the request the manager actually issued. Asserting that
a helper is *called* — or that its name appears in the method's source before
some other name — would pass just as happily if the transformed value were
discarded, if the sibling fields were dropped, or if no request went out at all.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from unifi_core.network.managers.client_group_manager import ClientGroupManager
from unifi_core.network.managers.firewall_manager import FirewallManager
from unifi_core.network.managers.network_manager import NetworkManager
from unifi_core.network.managers.stats_manager import StatsManager

UPPER = "AA:BB:CC:DD:EE:FF"
LOWER = "aa:bb:cc:dd:ee:ff"
UPPER_2 = "11:22:33:44:55:66"


class _RecordingConnection:
    """ConnectionManager stand-in that records requests and caches for real."""

    site = "default"

    def __init__(self, response: Any = None) -> None:
        self.requests: list[Any] = []
        self._cache: dict[str, Any] = {}
        self._response = response if response is not None else {"meta": {"rc": "ok"}}

    async def ensure_connected(self) -> bool:
        return True

    async def request(self, api_request: Any) -> Any:
        self.requests.append(api_request)
        return self._response

    def get_cached(self, key: str, timeout: int | None = None) -> Any:
        return self._cache.get(key)

    def _update_cache(self, key: str, value: Any, timeout: int | None = None) -> None:
        self._cache[key] = value

    def _invalidate_cache(self, *_args: Any) -> None:
        return None

    @property
    def sent(self) -> list[Any]:
        return [r.data for r in self.requests]

    @property
    def paths(self) -> list[str]:
        return [r.path for r in self.requests]


# --- client groups -----------------------------------------------------------


async def test_client_group_create_sends_lowercase_members_and_keeps_the_rest() -> None:
    connection = _RecordingConnection(response={"_id": "g1"})
    manager = ClientGroupManager(connection)

    await manager.create_client_group({"name": "Kids", "type": "CLIENTS", "members": [UPPER, UPPER_2]})

    assert connection.paths == ["/network-members-group"]
    sent = connection.sent[0]
    assert sent["members"] == [LOWER, "11:22:33:44:55:66"]
    assert sent["name"] == "Kids", "the group name was lost while normalising members"
    assert sent["type"] == "CLIENTS", "the group type was lost while normalising members"


# --- AP groups ---------------------------------------------------------------


async def test_ap_group_create_sends_lowercase_device_macs_and_keeps_the_rest() -> None:
    connection = _RecordingConnection(response={"_id": "ap1"})
    manager = NetworkManager(connection)

    await manager.create_ap_group({"name": "Upstairs", "device_macs": [UPPER], "attr_hidden_id": "x"})

    assert connection.paths == ["/apgroups"]
    sent = connection.sent[0]
    assert sent["device_macs"] == [LOWER]
    assert sent["name"] == "Upstairs"
    assert sent["attr_hidden_id"] == "x", "an unrelated field was dropped"


async def test_ap_group_update_normalizes_before_the_merge() -> None:
    """Order matters here, and the merged payload is where it shows.

    Normalising after the merge would leave the caller's uppercase restatement
    in the outgoing object; normalising after ``_unpersisted_fields`` inspects
    the update would make an unchanged-but-recased member list look like a
    field the controller refused to persist, reporting a failure for a write
    that succeeded.
    """
    connection = _RecordingConnection()
    manager = NetworkManager(connection)
    manager.get_ap_group_details = AsyncMock(
        return_value={"_id": "ap1", "name": "Upstairs", "device_macs": [LOWER], "site_id": "s1"}
    )

    await manager.update_ap_group("ap1", {"device_macs": [UPPER]})

    assert connection.paths == ["/apgroups/ap1"]
    sent = connection.sent[0]
    assert sent["device_macs"] == [LOWER], f"an uppercase MAC reached the controller: {sent['device_macs']}"
    assert sent["name"] == "Upstairs", "the merge lost a field the caller never mentioned"
    assert sent["site_id"] == "s1"


# --- firewall policies -------------------------------------------------------


async def test_firewall_create_lowercases_endpoint_macs_on_both_sides() -> None:
    connection = _RecordingConnection(response={"_id": "fp1"})
    manager = FirewallManager(connection)

    await manager.create_firewall_policy(
        {
            "name": "Block guest",
            "action": "BLOCK",
            "source": {"matching_target": "CLIENT", "client_macs": [UPPER]},
            "destination": {"matching_target": "ANY"},
        }
    )

    assert "/firewall-policies" in connection.paths
    sent = connection.sent[0]
    assert sent["source"]["client_macs"] == [LOWER]
    assert sent["source"]["matching_target"] == "CLIENT", "a sibling of client_macs was dropped"
    assert sent["destination"] == {"matching_target": "ANY"}, "the destination endpoint was mangled"
    assert sent["name"] == "Block guest"
    assert sent["action"] == "BLOCK"


# --- stats cache -------------------------------------------------------------


async def test_client_wifi_details_cache_is_case_stable() -> None:
    """A cache keyed on the caller's raw MAC fetches the same client twice.

    `/stat/sta` is a collection endpoint — it returns every connected wireless
    client — so a redundant miss is a full re-fetch, not a cheap one.
    """
    connection = _RecordingConnection(response=[{"mac": LOWER, "signal": -55, "channel": 36}])
    manager = StatsManager(connection, MagicMock())

    first = await manager.get_client_wifi_details(UPPER)
    second = await manager.get_client_wifi_details(LOWER)

    assert first is not None, "the uppercase lookup found nothing"
    assert second == first, "the two spellings disagreed about the same client"
    assert connection.paths == ["/stat/sta"], f"the controller was queried {len(connection.paths)} times for one client"
