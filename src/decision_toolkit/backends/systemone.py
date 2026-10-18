"""`decision_toolkit.backends.systemone` — talk to any System One server.

Ollama 0.35, vLLM, OpenJev's shim, OpenDecider, Von's server, LiteLLM or a
remote Jev endpoint all speak the same ``POST /v1/systemone`` request. This
backend is the toolkit's bridge to them (and to models too large for this
machine), using only the standard library so it never drags a dependency in.

Usage::

    DecisionModel(backend="systemone", backend_options={"base_url": "http://localhost:11434"})
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping

from ..errors import BackendError, BackendNotAvailableError
from ..models import ResolvedModel
from .base import BaseBackend


def _post_json(
    url: str, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None, timeout: float = 120
) -> Mapping[str, Any]:
    """POST JSON and decode the JSON response (isolated so tests can fake it)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **dict(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-provided URL
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise BackendError(f"HTTP {error.code} from {url}: {detail}") from error
    except urllib.error.URLError as error:
        raise BackendError(f"cannot reach {url}: {error.reason}") from error


class SystemOneBackend(BaseBackend):
    """A System One HTTP client: the server owns the model."""

    name = "systemone"

    def __init__(
        self,
        resolved: ResolvedModel,
        *,
        base_url: str = "",
        api_key: str | None = None,
        timeout: float = 120,
        model: str | None = None,
        device: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(resolved, device=device, **options)
        if not base_url:
            raise BackendNotAvailableError(
                "the 'systemone' backend talks to a running server; pass "
                "backend_options={'base_url': 'http://host:port'}"
            )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = float(timeout)
        self._model = model

    @property
    def url(self) -> str:
        return f"{self._base_url}/v1/systemone"

    def system_one(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        payload: dict[str, Any] = {"state": state, "questions": dict(questions)}
        if self._model:
            payload["model"] = self._model
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        data = _post_json(self.url, payload, headers=headers, timeout=self._timeout)
        answers = data.get("answers") if isinstance(data, Mapping) else None
        if not isinstance(answers, Mapping):
            raise BackendError(f"server at {self.url} returned no 'answers' object")
        return {str(key): dict(value) for key, value in answers.items()}


__all__ = ["SystemOneBackend"]
