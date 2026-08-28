"""A partial OON update must not rewrite fields the caller never mentioned.

`update_oon_policy` merges the caller's update over the policy already on the
controller and PUTs the whole object back. Anything the shaping step invents
therefore overwrites real configuration rather than defaulting a blank field,
and the tool's confirmation preview shows the caller's own input, not the
invented value - so the change is invisible until someone notices the policy
behaving differently.

These tests assert the complete outgoing request payload, because that is the
only place the defect is observable: the helper's return value looks reasonable
in isolation, and it is the merge that does the damage.
"""

from typing import Any
from unittest.mock import AsyncMock

from unifi_core.network.managers.oon_manager import OonManager

BLOCKLIST_POLICY: dict[str, Any] = {
    "_id": "policy-1",
    "name": "Guest restrictions",
    "enabled": True,
    "target_type": "CLIENTS",
    "targets": [{"type": "MAC", "value": "aa:bb:cc:dd:ee:ff"}],
    "secure": {
        "enabled": True,
        "internet": {"mode": "BLOCKLIST", "apps": ["some-app"]},
    },
}


class _RecordingConnection:
    """Minimal ConnectionManager stand-in that captures outgoing requests."""

    site = "default"

    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def ensure_connected(self) -> bool:
        return True

    async def request(self, api_request: Any) -> dict[str, Any]:
        self.requests.append(api_request)
        return {"meta": {"rc": "ok"}, "data": []}

    def _invalidate_cache(self, _prefix: str) -> None:
        return None


def _manager(existing: dict[str, Any]) -> tuple[OonManager, _RecordingConnection]:
    connection = _RecordingConnection()
    manager = OonManager(connection)
    manager.get_oon_policy_by_id = AsyncMock(return_value=existing)
    return manager, connection


async def test_disabling_secure_preserves_the_stored_internet_policy() -> None:
    """`{"secure": {"enabled": false}}` says one thing: stop enforcing.

    Shaping it as if it were a create filled in `internet.mode`, and because
    the update is merged over the existing policy, that invented mode replaced
    a stored BLOCKLIST with TURN_OFF_INTERNET - turning a narrow app block into
    a total internet cut for those clients.
    """
    manager, connection = _manager(BLOCKLIST_POLICY)

    await manager.update_oon_policy("policy-1", {"secure": {"enabled": False}})

    sent = connection.requests[0].data
    assert sent["secure"]["enabled"] is False, "the caller's own change must survive"
    assert sent["secure"]["internet"]["mode"] == "BLOCKLIST", (
        f"a partial update rewrote the stored internet mode into {sent['secure']['internet']['mode']!r}"
    )
    assert sent["secure"]["internet"]["apps"] == ["some-app"], "the stored app list was dropped"


async def test_renaming_a_policy_touches_nothing_else() -> None:
    """The narrowest possible update: the whole rest of the object must arrive
    back byte-for-byte as it was read."""
    manager, connection = _manager(BLOCKLIST_POLICY)

    await manager.update_oon_policy("policy-1", {"name": "Renamed"})

    sent = connection.requests[0].data
    assert sent["name"] == "Renamed"
    assert sent["secure"] == BLOCKLIST_POLICY["secure"]
    assert sent["targets"] == BLOCKLIST_POLICY["targets"]
    assert sent["enabled"] is True


async def test_a_caller_supplied_internet_mode_is_still_honoured() -> None:
    """Preserving omitted fields must not stop a caller changing one on
    purpose."""
    manager, connection = _manager(BLOCKLIST_POLICY)

    await manager.update_oon_policy("policy-1", {"secure": {"internet": {"mode": "TURN_OFF_INTERNET"}}})

    sent = connection.requests[0].data
    assert sent["secure"]["internet"]["mode"] == "TURN_OFF_INTERNET"


async def test_targets_are_still_normalized_on_the_way_out() -> None:
    """The wiring this PR added, asserted by its effect on the request rather
    than by reading the method's source: an uppercase MAC and a bare string
    both have to reach the controller in its own shape."""
    manager, connection = _manager(BLOCKLIST_POLICY)

    await manager.update_oon_policy("policy-1", {"targets": ["AA:BB:CC:DD:EE:11"]})

    sent = connection.requests[0].data
    assert sent["targets"] == [{"type": "MAC", "value": "aa:bb:cc:dd:ee:11"}]
