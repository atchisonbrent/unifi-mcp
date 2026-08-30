import asyncio
import os
import sys
from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class TransportStarted(RuntimeError):
    """Stop the server after proving startup reached the transport layer."""


@pytest.mark.asyncio
async def test_main_starts_mcp_transport_without_authenticating_controller(monkeypatch):
    """MCP readiness must not perform an externally visible controller login."""
    import unifi_mcp_shared.bootstrap as shared_bootstrap
    import unifi_mcp_shared.server_lifecycle as server_lifecycle
    import unifi_mcp_shared.tool_registration as tool_registration
    import unifi_mcp_shared.transport as transport
    import unifi_network_mcp.main as main
    import unifi_network_mcp.runtime as runtime

    initialize = AsyncMock(return_value=True)
    register_tools = AsyncMock()
    run_transports = AsyncMock(side_effect=TransportStarted)

    monkeypatch.setattr(runtime.connection_manager, "initialize", initialize)
    monkeypatch.setattr(shared_bootstrap, "assert_credentials_configured", lambda *args, **kwargs: None)
    monkeypatch.setattr(server_lifecycle, "install_asyncio_exception_handler", lambda _logger: None)
    monkeypatch.setattr(server_lifecycle, "apply_log_level", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tool_registration, "register_tools_for_mode", register_tools)
    monkeypatch.setattr(
        transport, "resolve_http_config", lambda *_args, **_kwargs: (False, "streamable-http", "127.0.0.1", 3000)
    )
    monkeypatch.setattr(transport, "run_transports", run_transports)

    with pytest.raises(TransportStarted):
        await main.main_async()

    initialize.assert_not_awaited()
    register_tools.assert_awaited_once()
    run_transports.assert_awaited_once()


@pytest.mark.asyncio
async def test_stdio_initialize_and_tool_discovery_make_no_controller_connection():
    """A real stdio handshake must be side-effect free toward the controller."""
    connection_count = 0

    async def record_connection(reader, writer):
        nonlocal connection_count
        connection_count += 1
        writer.close()
        await writer.wait_closed()

    sentinel = await asyncio.start_server(record_connection, "127.0.0.1", 0)
    port = sentinel.sockets[0].getsockname()[1]
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "unifi_network_mcp.main"],
        env={
            **os.environ,
            "UNIFI_HOST": "127.0.0.1",
            "UNIFI_PORT": str(port),
            "UNIFI_USERNAME": "startup-test",
            "UNIFI_PASSWORD": "startup-test",
            "UNIFI_CONTROLLER_TYPE": "proxy",
            "UNIFI_MCP_HTTP_ENABLED": "false",
            "UNIFI_TOOL_REGISTRATION_MODE": "lazy",
        },
    )

    try:
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
    finally:
        sentinel.close()
        await sentinel.wait_closed()

    assert tools.tools
    assert connection_count == 0
