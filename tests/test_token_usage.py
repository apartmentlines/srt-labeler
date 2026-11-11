"""Verify reported usage survives text conversion, failures and offline rescoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
import pytest

from srt_labeler.evaluate import Evaluation
from srt_labeler.labeling import LabelingEngine
from srt_labeler.token_usage import TokenUsage


def message() -> AIMessage:
    """Supply realistic totals with overlapping detail counters."""
    return AIMessage(
        content="response",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
            "input_token_details": {"audio": 80, "cache_read": 20},
            "output_token_details": {"reasoning": 10, "audio": 0},
        },
    )


def test_callback_captures_before_backend_strips_text(tmp_path: Path) -> None:
    model = FakeMessagesListChatModel(responses=[message()])
    backend = Mock()

    def run(*_args: Any, **_kwargs: Any) -> tuple[bool, str, None]:
        response = model.invoke("input")
        return True, str(response.content), None

    backend.run_template.side_effect = run
    with TokenUsage.capture(tmp_path / "usage.json") as usage:
        assert (
            LabelingEngine(backend).run_template("template", {}, usage=usage)[1]
            == "response"
        )
    report = json.loads((tmp_path / "usage.json").read_text())
    assert len(report["observations"]) == 1
    tokens = report["observations"][0]["tokens"]
    assert tokens["input_tokens"] == 100
    assert tokens["output_tokens"] == 25
    assert tokens["input_audio_tokens"] == 80
    assert tokens["cached_input_tokens"] == 20
    assert tokens["reasoning_tokens"] == 10
    assert tokens["output_audio_tokens"] == 0
    assert report["observations"][0]["raw"]["usage_metadata"]["total_tokens"] == 125
    model.invoke("outside scope")
    assert len(usage.observations) == 1


def test_callback_deduplication_and_fallback() -> None:
    usage = TokenUsage()
    result = LLMResult(generations=[[ChatGeneration(message=message())]])
    run_id = uuid4()
    usage.on_llm_end(result, run_id=run_id)
    usage.on_llm_end(result, run_id=run_id)
    usage.fallback(message())
    assert usage.report()["summary"]["metrics"]["input_tokens"]["total"] == 100
    assert len(usage.observations) == 1


def test_provider_fallback_preserves_raw_and_missing_fields() -> None:
    raw = {
        "prompt_tokens": 500,
        "completion_tokens": 30,
        "prompt_tokens_details": {"audio_tokens": 400},
    }
    usage = TokenUsage()
    usage.fallback(AIMessage(content="text", response_metadata={"token_usage": raw}))
    observation = usage.report()["observations"][0]
    assert observation["raw"]["provider_usage"] == raw
    assert observation["tokens"]["input_audio_tokens"] == 400
    assert observation["tokens"]["total_tokens"] is None
    assert observation["tokens"]["reasoning_tokens"] is None


def test_partial_totals_are_not_presented_as_complete(tmp_path: Path) -> None:
    usage = TokenUsage()
    usage.fallback(message())
    usage.record(usage.extract(None))
    metrics = usage.report()["summary"]["metrics"]["input_tokens"]
    assert metrics == {
        "total": None,
        "reported_subtotal": 100,
        "reported_observations": 1,
        "missing_observations": 1,
    }
    assert (
        Evaluation.read_usage(tmp_path / "absent.json")["summary"]["metrics"][
            "input_tokens"
        ]["total"]
        is None
    )


def test_failure_persists_usage_and_resets_scope(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        with TokenUsage.capture(tmp_path / "usage.json") as usage:
            usage.fallback(message())
            raise RuntimeError("later formatting failed")
    assert (
        Evaluation.read_usage(tmp_path / "usage.json")["summary"]["metrics"][
            "output_tokens"
        ]["total"]
        == 25
    )
    with TokenUsage.capture(tmp_path / "next.json"):
        pass
    assert (
        Evaluation.read_usage(tmp_path / "next.json")["summary"]["metrics"][
            "output_tokens"
        ]["total"]
        is None
    )
