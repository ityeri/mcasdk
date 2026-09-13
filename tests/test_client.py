from dataclasses import dataclass, field

import pytest

from miniviki.mca import (
    APPROVAL_REQUEST,
    MESSAGE,
    RUN_END,
    ClientCapability,
    ClientTool,
    ContextInit,
    MCAError,
    StreamEvent,
)
from miniviki.mcasdk import MiniVikiClient


@dataclass(slots=True)
class FakeTransport:
    events: list[StreamEvent] = field(default_factory=list)
    created: list[dict] = field(default_factory=list)
    submitted: list[tuple[str, str]] = field(default_factory=list)
    approvals: list[tuple[str, str, str]] = field(default_factory=list)
    tool_updates: list[tuple[str, list[ClientTool]]] = field(default_factory=list)
    interrupts: list[str] = field(default_factory=list)
    handle: dict = field(
        default_factory=lambda: {"id": "ctx_1", "kind": "main", "toolset_version": "v1"}
    )

    async def create_context(self, init: ContextInit) -> dict:
        self.created.append(init.to_json())
        return dict(self.handle)

    async def context_state(self, context_id: str) -> dict:
        return dict(self.handle) | {"id": context_id}

    async def submit(self, context_id: str, text: str) -> dict:
        self.submitted.append((context_id, text))
        return {"run_id": "run_1"}

    async def subscribe(self, context_id: str, from_seq: int = 0):
        for event in self.events:
            if event.seq >= from_seq:
                yield event

    async def interrupt(self, context_id: str) -> dict:
        self.interrupts.append(context_id)
        return {"ok": True}

    async def resolve_approval(self, context_id: str, call_id: str, decision: str) -> dict:
        self.approvals.append((context_id, call_id, decision))
        return {"ok": True}

    async def update_tools(self, context_id: str, tools: list[ClientTool]) -> dict:
        self.tool_updates.append((context_id, tools))
        return dict(self.handle) | {"toolset_version": "v2"}

    async def aclose(self) -> None:
        return None


def event(seq: int, kind: str, **payload) -> StreamEvent:
    return StreamEvent(seq=seq, kind=kind, payload=payload)


def build_client(**kwargs) -> tuple[MiniVikiClient, FakeTransport]:
    transport = FakeTransport(**kwargs)
    return MiniVikiClient(transport=transport), transport


async def test_open_sends_capabilities_and_local_tools():
    client, transport = build_client()
    client.capabilities = ClientCapability(label="ramyon", shell=True, approval_ui=True)
    client.tools = (ClientTool(name="client:shell", description="run locally"),)
    handle = await client.open()
    assert handle.id == "ctx_1"
    sent = transport.created[0]
    assert sent["capabilities"]["label"] == "ramyon"
    assert sent["capabilities"]["shell"] is True
    assert sent["tools"][0]["name"] == "client:shell"


async def test_ask_returns_the_assistant_text_and_stops_at_run_end():
    client, transport = build_client(
        events=[
            event(0, MESSAGE, role="user", content="hi"),
            event(1, MESSAGE, role="assistant", content="hello "),
            event(2, MESSAGE, role="assistant", content="world"),
            event(3, RUN_END, status="done")
        ]
    )
    await client.open()
    turn = await client.ask("hi")
    assert turn.text == "hello world"
    assert turn.status == "done"
    assert transport.submitted == [("ctx_1", "hi")]


async def test_ask_reports_a_parked_approval():
    client, _transport = build_client(
        events=[event(0, APPROVAL_REQUEST, status="waiting_approval", tool="exec", call_id="c1")]
    )
    await client.open()
    turn = await client.ask("clean up")
    assert turn.needs_approval
    assert turn.status == "waiting_approval"


async def test_resume_continues_after_the_last_seen_event():
    client, transport = build_client(events=[event(0, MESSAGE, role="assistant", content="first")])
    await client.open()
    await client.ask("hi")
    assert client.last_seq == 0
    transport.events = [
        event(0, MESSAGE, role="assistant", content="first"),
        event(1, MESSAGE, role="assistant", content="second"),
        event(2, RUN_END, status="done")
    ]
    turn = await client.resume()
    assert turn.text == "second"
    assert turn.status == "done"


async def test_approve_sends_the_decision_and_collects_the_outcome():
    parked = event(0, APPROVAL_REQUEST, status="waiting_approval", call_id="c1")
    client, transport = build_client(events=[parked])
    await client.open()
    await client.ask("go")
    transport.events = [
        event(0, APPROVAL_REQUEST, status="waiting_approval", call_id="c1"),
        event(1, MESSAGE, role="assistant", content="ran it"),
        event(2, RUN_END, status="done")
    ]
    turn = await client.approve("c1")
    assert transport.approvals == [("ctx_1", "c1", "approved")]
    assert turn.text == "ran it"


async def test_deny_sends_a_denial():
    parked = event(0, APPROVAL_REQUEST, status="waiting_approval", call_id="c7")
    client, transport = build_client(events=[parked])
    await client.open()
    await client.ask("go")
    transport.events = [event(0, RUN_END, status="done")]
    await client.deny("c7")
    assert transport.approvals == [("ctx_1", "c7", "denied")]


async def test_set_tools_pushes_client_tools_and_refreshes_the_toolset_version():
    client, transport = build_client()
    await client.open()
    handle = await client.set_tools([ClientTool(name="client:read_attachment")])
    assert handle.toolset_version == "v2"
    assert transport.tool_updates[0][1][0].name == "client:read_attachment"


async def test_interrupt_reaches_the_transport():
    client, transport = build_client()
    await client.open()
    await client.interrupt()
    assert transport.interrupts == ["ctx_1"]


async def test_attach_reuses_an_existing_context():
    client, _transport = build_client()
    handle = await client.attach("ctx_9")
    assert handle.id == "ctx_9"
    assert client.handle is not None


async def test_using_the_client_without_a_context_is_an_error():
    client, _transport = build_client()
    with pytest.raises(MCAError):
        await client.ask("hi")


async def test_collection_stops_at_the_event_ceiling():
    client, _transport = build_client(
        events=[event(index, MESSAGE, role="assistant", content="x") for index in range(10)]
    )
    client.max_events = 4
    await client.open()
    turn = await client.ask("hi")
    assert turn.status == "truncated"
    assert len(turn.events) == 4


async def test_capability_flags_round_trip_through_json():
    capability = ClientCapability(label="web", shell=True, approval_ui=True)
    restored = ClientCapability.from_json(capability.to_json())
    assert restored == capability
    assert "shell" in capability.flags()
