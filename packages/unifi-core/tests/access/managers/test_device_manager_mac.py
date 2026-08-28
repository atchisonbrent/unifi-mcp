"""Access device lookups accept a MAC in either case, and resolve it.

`device_id` here may be either an opaque `unique_id` or a MAC. Only the MAC
arm is case-insensitive: unique_ids are not addresses, so relaxing their
comparison would be a guess rather than a fix.

Resolving matters as much as matching. The reboot path interpolates its
argument straight into the request path, so a lookup that starts accepting
a MAC must hand the controller the identifier the controller actually
indexes by, not the string the caller happened to type.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from unifi_core.access.managers.device_manager import DeviceManager

LOWER = "aa:bb:cc:dd:ee:ff"
UPPER = "AA:BB:CC:DD:EE:FF"
UNIQUE_ID = "0123456789abcdef01234567"
OTHER_ID = "fedcba9876543210fedcba98"
OTHER_MAC = "11:22:33:44:55:66"

# topology4 nests devices as site -> floors -> doors -> device_groups -> devices
TOPOLOGY = {
    "data": [
        {
            "floors": [
                {
                    "doors": [
                        {
                            "name": "Entry",
                            "unique_id": "door-1",
                            "device_groups": [
                                [
                                    {"unique_id": UNIQUE_ID, "mac": LOWER, "name": "Entry Reader", "type": "UA-G2"},
                                    {"unique_id": OTHER_ID, "mac": OTHER_MAC, "name": "Side Reader", "type": "UA-G2"},
                                ]
                            ],
                        }
                    ]
                }
            ]
        }
    ]
}


def _manager() -> DeviceManager:
    cm = MagicMock()
    cm.has_api_client = False  # force the proxy path, which is the one that matches on MAC
    cm.has_proxy = True
    cm.proxy_request = AsyncMock(return_value=TOPOLOGY)
    cm.extract_data = MagicMock(side_effect=lambda d: d.get("data", []))
    return DeviceManager(cm)


@pytest.mark.asyncio
async def test_get_device_accepts_an_uppercase_mac() -> None:
    mgr = _manager()
    assert (await mgr.get_device(UPPER))["name"] == "Entry Reader"


@pytest.mark.asyncio
async def test_get_device_still_matches_a_unique_id_exactly() -> None:
    """With two devices present, a match-anything predicate returns the wrong
    one - which a single-device fixture could never detect."""
    mgr = _manager()
    assert (await mgr.get_device(OTHER_ID))["name"] == "Side Reader"
    assert (await mgr.get_device(UNIQUE_ID))["name"] == "Entry Reader"


@pytest.mark.asyncio
async def test_get_device_raises_for_an_unknown_identifier() -> None:
    from unifi_core.exceptions import UniFiNotFoundError

    mgr = _manager()
    with pytest.raises(UniFiNotFoundError):
        await mgr.get_device("99:99:99:99:99:99")


@pytest.mark.asyncio
async def test_reboot_preview_resolves_a_mac_to_the_unique_id() -> None:
    """The preview's device_id is what the confirm step sends to the URL."""
    mgr = _manager()
    assert (await mgr.reboot_device(UPPER))["device_id"] == UNIQUE_ID


@pytest.mark.asyncio
async def test_apply_reboot_posts_to_the_unique_id_not_the_callers_mac() -> None:
    mgr = _manager()
    await mgr.apply_reboot_device(UPPER)
    paths = [c.args[1] for c in mgr._cm.proxy_request.call_args_list if len(c.args) > 1]
    assert f"devices/{UNIQUE_ID}/reboot" in paths, paths
    assert not any(UPPER in p for p in paths), f"the caller's raw MAC reached the URL: {paths}"


# --- dual auth: the public API path ------------------------------------------
#
# The tests above force `has_api_client = False` so they exercise the proxy,
# which is the arm that matches on MAC. That left the API arm untested, and it
# is a different shape entirely: `py-unifi-access`'s `Device` carries the MAC
# in `id` and has no `mac` attribute at all, so matching `mac_equal` against
# `.mac` compares against None on every row.

Device = pytest.importorskip("unifi_access_api").Device


def _dual_auth_manager() -> DeviceManager:
    """A controller answering on BOTH paths, which is the normal deployment."""
    cm = MagicMock()
    cm.has_api_client = True
    cm.has_proxy = True
    cm.api_client = MagicMock()
    cm.api_client.get_devices = AsyncMock(
        return_value=[
            Device(id=LOWER, name="Entry Reader", type="UA-G2", is_online=True),
            Device(id=OTHER_MAC, name="Side Reader", type="UA-G2", is_online=True),
        ]
    )
    cm.proxy_request = AsyncMock(return_value=TOPOLOGY)
    cm.extract_data = MagicMock(side_effect=lambda d: d.get("data", []))
    return DeviceManager(cm)


def test_the_real_device_model_has_no_mac_attribute() -> None:
    """The premise, pinned: if the dependency ever grows a `mac`, the mapping
    below should be revisited rather than silently kept."""
    device = Device(id=LOWER)
    assert not hasattr(device, "mac") or getattr(device, "mac", None) is None
    assert device.id == LOWER


@pytest.mark.asyncio
async def test_api_path_accepts_an_uppercase_mac() -> None:
    """`Device.id` IS the MAC, so it has to be compared as one."""
    mgr = _dual_auth_manager()

    device = await mgr.get_device(UPPER)

    assert device["name"] == "Entry Reader"


@pytest.mark.asyncio
async def test_api_path_reports_the_id_as_a_mac() -> None:
    """Labelling the MAC as `unique_id` invites the reboot path to post it as
    the controller's own identifier, which is a different namespace."""
    mgr = _dual_auth_manager()

    device = await mgr.get_device(LOWER)

    assert device["mac"] == LOWER, "the MAC was reported as absent when it was sitting in `id`"
    assert device.get("unique_id") != LOWER, "a MAC was presented as the controller's unique_id"


@pytest.mark.asyncio
async def test_reboot_resolves_the_controller_identifier_under_dual_auth() -> None:
    """The reboot target must be the identifier the controller indexes by.

    The API path knows the device by MAC and the controller's reboot endpoint
    does not, so the MAC has to be resolved through the proxy topology even
    though the API client answered the lookup.
    """
    mgr = _dual_auth_manager()

    preview = await mgr.reboot_device(UPPER)

    assert preview["device_id"] == UNIQUE_ID, f"reboot would target {preview['device_id']!r}, which is not a unique_id"
    assert preview["device_name"] == "Entry Reader"


# --- the shape a real controller actually returns -----------------------------
#
# The fixtures above write the MAC the same way on both sides, which is what
# hid this: on a live controller the API client reports `Device.id` as
# "1c0b8beef6b5" while the topology payload writes the same device
# "1c:0b:8b:ee:f6:b5". A case-only comparison treats those as two devices, so
# the reboot resolution found nothing and silently fell back to posting the
# address.

BARE_MAC = "1c0b8beef6b5"
SEPARATED_MAC = "1c:0b:8b:ee:f6:b5"
HUB_UNIQUE_ID = "89abcdef0123456789abcdef"

MIXED_FORM_TOPOLOGY = {
    "data": [
        {
            "floors": [
                {
                    "doors": [
                        {
                            "name": "Entry",
                            "unique_id": "door-1",
                            "device_groups": [
                                [{"unique_id": HUB_UNIQUE_ID, "mac": SEPARATED_MAC, "name": "Hub", "type": "UA-Hub"}]
                            ],
                        }
                    ]
                }
            ]
        }
    ]
}


def _mixed_form_manager() -> DeviceManager:
    cm = MagicMock()
    cm.has_api_client = True
    cm.has_proxy = True
    cm.api_client = MagicMock()
    cm.api_client.get_devices = AsyncMock(return_value=[Device(id=BARE_MAC, name="Hub", type="UA-Hub", is_online=True)])
    cm.proxy_request = AsyncMock(return_value=MIXED_FORM_TOPOLOGY)
    cm.extract_data = MagicMock(side_effect=lambda d: d.get("data", []))
    return DeviceManager(cm)


@pytest.mark.asyncio
async def test_reboot_resolves_across_separator_forms() -> None:
    """The API arm answers with the bare form; the topology holds the
    separated one. Both name the same hub, so the reboot must target its
    unique_id."""
    mgr = _mixed_form_manager()

    preview = await mgr.reboot_device(BARE_MAC.upper())

    assert preview["device_id"] == HUB_UNIQUE_ID, (
        f"reboot would post {preview['device_id']!r}, which is an address rather than a unique_id"
    )


@pytest.mark.asyncio
async def test_lookup_accepts_either_separator_form() -> None:
    mgr = _mixed_form_manager()

    assert (await mgr.get_device(SEPARATED_MAC))["name"] == "Hub"
    assert (await mgr.get_device(BARE_MAC))["name"] == "Hub"
