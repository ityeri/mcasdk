import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from miniviki.mca import ClientTool, ContextInit, StreamEvent, TransportError


@dataclass(slots=True)
class HttpTransport:
    """http + server-sent events. Resume is a query parameter, not a new session."""

    base_url: str
    token: str = ""
    timeout: float = 60.0
    http: httpx.AsyncClient = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.http is None:
            self.http = httpx.AsyncClient(timeout=self.timeout)

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def _json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        response = await self.http.request(
            method, self._url(path), headers=self._headers(), json=body
        )
        if response.status_code >= 400:
            raise TransportError(
                f"{method} {path} -> {response.status_code}: {response.text[:200]}"
            )
        return dict(response.json())

    async def create_context(self, init: ContextInit) -> dict[str, Any]:
        return await self._json("POST", "/contexts", init.to_json())

    async def context_state(self, context_id: str) -> dict[str, Any]:
        return await self._json("GET", f"/contexts/{context_id}")

    async def submit(self, context_id: str, text: str) -> dict[str, Any]:
        return await self._json("POST", f"/contexts/{context_id}/input", {"text": text})

    async def interrupt(self, context_id: str) -> dict[str, Any]:
        return await self._json("POST", f"/contexts/{context_id}/interrupt")

    async def resolve_approval(
        self,
        context_id: str,
        call_id: str,
        decision: str
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"/contexts/{context_id}/approvals",
            {"call_id": call_id, "decision": decision}
        )

    async def update_tools(self, context_id: str, tools: list[ClientTool]) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"/contexts/{context_id}/tools",
            {"tools": [tool.to_json() for tool in tools]}
        )

    async def subscribe(self, context_id: str, from_seq: int = 0) -> AsyncIterator[StreamEvent]:
        async with self.http.stream(
            "GET",
            self._url(f"/contexts/{context_id}/events"),
            headers=self._headers(),
            params={"from_seq": from_seq}
        ) as response:
            if response.status_code >= 400:
                raise TransportError(f"subscribe -> {response.status_code}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if not body:
                    continue
                yield StreamEvent.from_json(json.loads(body))

    async def aclose(self) -> None:
        await self.http.aclose()
