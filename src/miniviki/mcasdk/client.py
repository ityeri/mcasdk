from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from miniviki.mca import (
    MESSAGE,
    ClientCapability,
    ClientRequest,
    ClientTool,
    ContextHandle,
    ContextInit,
    MCAError,
    StreamEvent,
    Transport,
    Turn,
)

# What a client registers for a tool it declared: arguments in, text out.
type ClientToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(slots=True)
class MiniVikiClient:
    """The code level API. A front depends on this and nothing else.

    Any object satisfying the `Transport` contract works here, including one you
    write yourself in a few lines -- http is the shipped one, not the only one.
    """

    transport: Transport
    capabilities: ClientCapability = field(default_factory=ClientCapability)
    tools: tuple[ClientTool, ...] = ()
    handlers: dict[str, ClientToolHandler] = field(default_factory=dict)
    handle: ContextHandle | None = None
    last_seq: int = -1
    max_events: int = 2048

    async def open(self, init: ContextInit | None = None) -> ContextHandle:
        payload = init or ContextInit(capabilities=self.capabilities, tools=self.tools)
        raw = await self.transport.create_context(payload)
        self.handle = ContextHandle.from_json(raw)
        return self.handle

    async def attach(self, context_id: str) -> ContextHandle:
        raw = await self.transport.context_state(context_id)
        self.handle = ContextHandle.from_json(raw)
        return self.handle

    async def ask(self, text: str, max_events: int | None = None) -> Turn:
        handle = self._require_handle()
        submitted = await self.transport.submit(handle.id, text)
        return await self._collect(submitted.get("run_id", ""), max_events or self.max_events)

    async def resume(self, max_events: int | None = None) -> Turn:
        return await self._collect("", max_events or self.max_events)

    async def interrupt(self) -> None:
        handle = self._require_handle()
        await self.transport.interrupt(handle.id)

    async def approve(self, call_id: str) -> Turn:
        return await self._decide(call_id, "approved")

    async def deny(self, call_id: str) -> Turn:
        return await self._decide(call_id, "denied")

    async def serve(self, request: ClientRequest) -> Turn:
        """Run one of this client's own tools and hand the result back.

        A request with no registered handler is answered as an error rather than
        left alone: the run is parked on this, so silence would wedge the context.
        """
        handler = self.handlers.get(request.client_tool)
        if handler is None:
            return await self.decline(
                request,
                f"error: this client registered no handler for {request.client_tool!r}"
            )
        try:
            content = await handler(request.arguments)
        except Exception as error:
            content = f"error: {request.tool} failed: {type(error).__name__}: {error}"
        return await self._report(request, content)

    async def decline(self, request: ClientRequest, reason: str) -> Turn:
        """Answer without running anything, so the model learns why it did not happen."""
        return await self._report(request, reason)

    async def set_tools(self, tools: list[ClientTool]) -> ContextHandle:
        handle = self._require_handle()
        self.tools = tuple(tools)
        raw = await self.transport.update_tools(handle.id, tools)
        self.handle = ContextHandle.from_json(raw)
        return self.handle

    def stream(self, from_seq: int | None = None) -> AsyncIterator[StreamEvent]:
        handle = self._require_handle()
        start = self.last_seq + 1 if from_seq is None else from_seq
        return self.transport.subscribe(handle.id, from_seq=start)

    async def aclose(self) -> None:
        await self.transport.aclose()

    async def _report(self, request: ClientRequest, content: str) -> Turn:
        handle = self._require_handle()
        await self.transport.report_tool_result(handle.id, request.call_id, content)
        return await self._collect("", self.max_events)

    async def _decide(self, call_id: str, decision: str) -> Turn:
        handle = self._require_handle()
        await self.transport.resolve_approval(handle.id, call_id, decision)
        return await self._collect("", self.max_events)

    async def _collect(self, run_id: str, max_events: int) -> Turn:
        chunks: list[str] = []
        events: list[StreamEvent] = []
        status = "unknown"
        async for event in self.stream():
            events.append(event)
            self.last_seq = max(self.last_seq, event.seq)
            if event.kind == MESSAGE:
                content = str(event.payload.get("content", ""))
                if event.payload.get("role") == "assistant" and content:
                    chunks.append(content)
            if event.is_terminal():
                status = str(event.payload.get("status", event.kind))
                break
            if len(events) >= max_events:
                status = "truncated"
                break
        return Turn(text="".join(chunks), status=status, events=tuple(events), run_id=run_id)

    def _require_handle(self) -> ContextHandle:
        if self.handle is None:
            raise MCAError("open() or attach() a context before using the client")
        return self.handle
