import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

import pytest

from miniviki.mca import ClientTool, ContextInit
from miniviki.mcasdk import HttpTransport

_SCRIPT = [
    {"seq": 0, "kind": "message", "payload": {"role": "assistant", "content": "hello "}},
    {"seq": 1, "kind": "message", "payload": {"role": "assistant", "content": "there"}},
    {"seq": 2, "kind": "run_end", "payload": {"status": "done"}}
]


class _Stub(BaseHTTPRequestHandler):
    seen: ClassVar[list[tuple[str, str, dict]]] = []

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        _Stub.seen.append((self.command, self.path, body))
        if self.path == "/contexts":
            self._json({"id": "ctx_1", "kind": body.get("kind", "main"), "toolset_version": "v1"})
        elif self.path.endswith("/input"):
            self._json({"run_id": "run_1"})
        elif self.path.endswith("/approvals"):
            self._json({"ok": True, "decision": body.get("decision")})
        elif self.path.endswith("/tools"):
            self._json({"id": "ctx_1", "kind": "main", "toolset_version": "v2"})
        elif self.path.endswith("/interrupt"):
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, status=404)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        _Stub.seen.append((self.command, parsed.path, {}))
        if parsed.path.endswith("/events"):
            start = int(parse_qs(parsed.query).get("from_seq", ["0"])[0])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for entry in _SCRIPT:
                if entry["seq"] >= start:
                    self.wfile.write(f"data: {json.dumps(entry)}\n\n".encode())
            self.wfile.flush()
            return
        self._json({"id": "ctx_1", "kind": "main", "toolset_version": "v1"})

    def log_message(self, *args) -> None:
        return None


@pytest.fixture
def stub():
    _Stub.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield HttpTransport(base_url=f"http://127.0.0.1:{server.server_port}")
    server.shutdown()
    thread.join(timeout=5)


async def test_create_context_posts_the_init_payload(stub):
    raw = await stub.create_context(ContextInit(kind="side", label="notes"))
    assert raw["id"] == "ctx_1"
    sent = _Stub.seen[-1][2]
    assert sent["kind"] == "side"
    assert sent["label"] == "notes"
    await stub.aclose()


async def test_submit_and_approval_and_tools_round_trip(stub):
    await stub.submit("ctx_1", "hello")
    assert _Stub.seen[-1][2] == {"text": "hello"}
    await stub.resolve_approval("ctx_1", "c1", "approved")
    assert _Stub.seen[-1][2] == {"call_id": "c1", "decision": "approved"}
    await stub.update_tools("ctx_1", [ClientTool(name="client:shell")])
    assert _Stub.seen[-1][2]["tools"][0]["name"] == "client:shell"
    await stub.aclose()


async def test_subscribe_parses_the_event_stream(stub):
    events = [event async for event in stub.subscribe("ctx_1", from_seq=0)]
    assert [event.seq for event in events] == [0, 1, 2]
    assert events[0].payload["content"] == "hello "
    assert events[-1].kind == "run_end"
    await stub.aclose()


async def test_subscribe_resumes_from_a_sequence_number(stub):
    events = [event async for event in stub.subscribe("ctx_1", from_seq=2)]
    assert [event.seq for event in events] == [2]
    assert _Stub.seen[-1][1].endswith("/events")
    await stub.aclose()


async def test_a_404_becomes_a_transport_error(stub):
    from miniviki.mca import TransportError

    with pytest.raises(TransportError):
        await stub._json("POST", "/nope")
    await stub.aclose()


async def test_client_over_the_http_transport_end_to_end(stub):
    from miniviki.mcasdk import MiniVikiClient

    client = MiniVikiClient(transport=stub)
    handle = await client.open()
    assert handle.id == "ctx_1"
    turn = await client.ask("hi")
    assert turn.text == "hello there"
    assert turn.status == "done"
    await client.aclose()
