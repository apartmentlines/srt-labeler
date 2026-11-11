"""Capture reported model usage before LWE converts attached-file responses to text."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
from threading import Lock
from typing import Any
from typing_extensions import override

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook

FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_audio_tokens",
    "output_audio_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)


class TokenUsage(BaseCallbackHandler):
    """Collect one usage observation per completed or failed model request."""

    def __init__(self) -> None:
        super().__init__()
        self.observations: list[dict[str, Any]] = []
        self.seen: set[str] = set()
        self.lock: Any = Lock()

    @staticmethod
    def normalize(raw: dict[str, Any]) -> dict[str, int | None]:
        """Prefer standard usage and retain missing breakdowns as null.

        Reasoning and cached/audio counts are subsets, never added to totals.
        """
        standard = raw.get("usage_metadata") or {}
        provider = raw.get("provider_usage") or {}

        def number(*values: Any) -> int | None:
            return next((v for v in values if type(v) is int and v >= 0), None)

        inputs = (
            standard.get("input_token_details")
            or provider.get("prompt_tokens_details")
            or provider.get("input_tokens_details")
            or {}
        )
        outputs = (
            standard.get("output_token_details")
            or provider.get("completion_tokens_details")
            or provider.get("output_tokens_details")
            or {}
        )
        return {
            "input_tokens": number(
                standard.get("input_tokens"),
                provider.get("prompt_tokens"),
                provider.get("input_tokens"),
                provider.get("promptTokenCount"),
                provider.get("prompt_token_count"),
            ),
            "output_tokens": number(
                standard.get("output_tokens"),
                provider.get("completion_tokens"),
                provider.get("output_tokens"),
            ),
            "total_tokens": number(
                standard.get("total_tokens"),
                provider.get("total_tokens"),
                provider.get("totalTokenCount"),
                provider.get("total_token_count"),
            ),
            "input_audio_tokens": number(
                inputs.get("audio"), inputs.get("audio_tokens")
            ),
            "output_audio_tokens": number(
                outputs.get("audio"), outputs.get("audio_tokens")
            ),
            "cached_input_tokens": number(
                inputs.get("cache_read"),
                inputs.get("cached_tokens"),
                provider.get("cachedContentTokenCount"),
                provider.get("cached_content_token_count"),
            ),
            "reasoning_tokens": number(
                outputs.get("reasoning"),
                outputs.get("reasoning_tokens"),
                provider.get("thoughtsTokenCount"),
                provider.get("thoughts_token_count"),
            ),
        }

    @staticmethod
    def extract(
        response: Any, llm_output: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Retain only usage metadata, excluding response text and request secrets."""
        metadata = response.response_metadata if isinstance(response, AIMessage) else {}
        standard = response.usage_metadata if isinstance(response, AIMessage) else None
        output = llm_output or {}
        raw = {
            "usage_metadata": standard,
            "provider_usage": metadata.get("token_usage")
            or metadata.get("usage_metadata")
            or metadata.get("usage")
            or output.get("token_usage")
            or output.get("usage"),
        }
        return {"raw": raw, "tokens": TokenUsage.normalize(raw)}

    @override
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Capture the final aggregated response once, not individual stream chunks."""
        message = None
        if response.generations and response.generations[0]:
            generation = response.generations[0][0]
            if isinstance(generation, ChatGeneration):
                message = generation.message
        self.record(self.extract(message, response.llm_output), kwargs.get("run_id"))

    @override
    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Count failed requests without inventing unreported usage."""
        response = kwargs.get("response")
        if isinstance(response, LLMResult):
            self.on_llm_end(
                response, **{k: v for k, v in kwargs.items() if k != "response"}
            )
        else:
            self.record(self.extract(None), kwargs.get("run_id"))

    def record(self, observation: dict[str, Any], run_id: Any = None) -> None:
        """Deduplicate callback events for the same request."""
        with self.lock:
            if run_id is not None and str(run_id) in self.seen:
                return
            if run_id is not None:
                self.seen.add(str(run_id))
            self.observations.append(observation)

    def fallback(self, response: Any) -> None:
        """Use a returned message only when no callback captured this run."""
        if not self.observations:
            self.record(self.extract(response))

    @staticmethod
    def aggregate(observations: list[dict[str, Any]]) -> dict[str, Any]:
        """Report complete totals separately from partial reported subtotals."""
        metrics = {}
        for field in FIELDS:
            values = [o.get("tokens", {}).get(field) for o in observations]
            known = [v for v in values if type(v) is int and v >= 0]
            metrics[field] = {
                "total": sum(known) if known and len(known) == len(values) else None,
                "reported_subtotal": sum(known) if known else None,
                "reported_observations": len(known),
                "missing_observations": len(values) - len(known),
            }
        return {"observations": len(observations), "metrics": metrics}

    def report(self) -> dict[str, Any]:
        """Represent missing observations explicitly, including interrupted calls."""
        observations = self.observations or [self.extract(None)]
        return {
            "schema_version": 1,
            "source": "provider-reported",
            "observations": observations,
            "summary": self.aggregate(observations),
        }

    @classmethod
    @contextmanager
    def capture(cls, destination: Path) -> Generator[TokenUsage, None, None]:
        """Scope callbacks to one CLI attempt and persist usage even on failure."""
        collector = cls()
        token = USAGE_CONTEXT.set(collector)
        try:
            yield collector
        finally:
            USAGE_CONTEXT.reset(token)
            destination.write_text(
                json.dumps(collector.report(), indent=2), encoding="utf-8"
            )


USAGE_CONTEXT: ContextVar[TokenUsage | None] = ContextVar(
    "srt_labeler_usage", default=None
)
register_configure_hook(USAGE_CONTEXT, inheritable=True)
