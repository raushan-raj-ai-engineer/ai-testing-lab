from __future__ import annotations
from typing import Any
from src.observability.tracer import TraceRecorder

class TimedMCPClient:
    def __init__(self, client: Any, tracer: TraceRecorder) -> None:
        self._client = client; self.tracer = tracer
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        args = arguments or {}
        with self.tracer.span(f'mcp.{name}', 'mcp_tool', {'tool_name': name, 'argument_keys': sorted(args.keys())}) as span:
            result = await self._client.call_tool(name, args)
            is_error = bool(getattr(result, 'is_error', False)); span.set_attribute('mcp_is_error', is_error)
            if is_error: span.mark_error('MCP tool returned is_error=True')
            return result
    async def list_tools(self) -> Any:
        with self.tracer.span('mcp.list_tools', 'mcp_discovery'):
            return await self._client.list_tools()
    def __getattr__(self, name: str) -> Any: return getattr(self._client, name)
