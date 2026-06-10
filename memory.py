"""Incremental run traces and post-task memory generation."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Template

from conversation import Convo


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / ".runs"
MEMORIES_DIR = ROOT / ".memories"
MEMORY_PROMPT_PATH = ROOT / "memory_prompt.j2"
MEMORY_SCHEMA_VERSION = 2
DEFAULT_VERIFICATION_CONFIDENCE = 0.75

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slugify(text: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "task")[:max_length].rstrip("-")


def _new_run_id(task: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{timestamp}_{_slugify(task)}_{uuid.uuid4().hex[:8]}"


class RunRecorder:
    """Append factual events as JSONL so a partial run survives a crash."""

    def __init__(self, task: str, *, directory: Path = RUNS_DIR) -> None:
        self.task = task
        self.run_id = _new_run_id(task)
        self.started_at = utc_now()
        self.trajectory: list[dict[str, Any]] = []
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.run_id}.jsonl"
        self._append(
            {
                "event": "run_started",
                "run_id": self.run_id,
                "task": task,
                "started_at": self.started_at,
            }
        )

    def record_initial_observation(
        self,
        screenshot: dict[str, Any],
        mouse: dict[str, Any],
        error: str | None,
    ) -> None:
        self._append(
            {
                "event": "initial_observation",
                "recorded_at": utc_now(),
                "screenshot": screenshot,
                "mouse": mouse,
                "observation_error": error,
            }
        )

    def record_step(self, record: dict[str, Any]) -> None:
        factual_record = dict(record)
        factual_record.setdefault("recorded_at", utc_now())
        self.trajectory.append(factual_record)
        self._append({"event": "step", **factual_record})

    def finish(self, status: str, **details: Any) -> None:
        self._append(
            {
                "event": "run_finished",
                "finished_at": utc_now(),
                "status": status,
                **details,
            }
        )

    def _append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str))
            file.write("\n")
            file.flush()


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(stripped[start : end + 1])

    if not isinstance(data, dict):
        raise TypeError("memory review response must be a JSON object.")
    return data


def _number(value: Any, default: float = 0.0) -> float:
    if not isinstance(value, int | float):
        return default
    return max(0.0, min(1.0, float(value)))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _target_mappings(value: Any) -> list[dict[str, Any]]:
    mappings = []
    for item in _dict_list(value)[:3]:
        mappings.append(
            {
                "requested": str(item.get("requested", "")),
                "effective": str(item.get("effective", "")),
                "relationship": str(item.get("relationship", "")),
                "confidence": _number(item.get("confidence")),
                "revalidate_next_time": bool(item.get("revalidate_next_time", True)),
            }
        )
    return mappings


def _plan_steps(value: Any) -> list[dict[str, Any]]:
    steps = []
    for item in _dict_list(value)[:4]:
        steps.append(
            {
                "intent": str(item.get("intent", "")),
                "target": str(item.get("target", "")),
                "method": str(item.get("method", "")),
                "preconditions": _string_list(item.get("preconditions"))[:3],
                "success_check": str(item.get("success_check", "")),
            }
        )
    return steps


def _fallbacks(value: Any) -> list[dict[str, Any]]:
    fallbacks = []
    for item in _dict_list(value)[:2]:
        fallbacks.append(
            {
                "when": str(item.get("when", "")),
                "method": str(item.get("method", "")),
                "success_check": str(item.get("success_check", "")),
            }
        )
    return fallbacks


def _environment_facts(value: Any) -> list[dict[str, Any]]:
    facts = []
    for item in _dict_list(value)[:3]:
        facts.append(
            {
                "fact": str(item.get("fact", "")),
                "confidence": _number(item.get("confidence")),
                "revalidate_next_time": bool(item.get("revalidate_next_time", True)),
            }
        )
    return facts


def normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    verification = review.get("verification")
    if not isinstance(verification, dict):
        verification = {}

    playbook = review.get("playbook")
    if not isinstance(playbook, dict):
        playbook = {}

    return {
        "verification": {
            "success": bool(verification.get("success", False)),
            "confidence": _number(verification.get("confidence")),
            "summary": str(verification.get("summary", "")),
        },
        "playbook": {
            "task_signature": str(playbook.get("task_signature", "")),
            "applicability": str(playbook.get("applicability", "")),
            "learned_target_mappings": _target_mappings(
                playbook.get("learned_target_mappings")
            ),
            "preferred_plan": _plan_steps(playbook.get("preferred_plan")),
            "fallbacks": _fallbacks(playbook.get("fallbacks")),
            "avoid": _string_list(playbook.get("avoid"))[:3],
            "timing_notes": _string_list(playbook.get("timing_notes"))[:2],
            "environment_facts": _environment_facts(playbook.get("environment_facts")),
        },
    }


def _review_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate screenshot metadata before the post-task model call."""

    compact = []
    for record in trajectory:
        reply = record.get("reply")
        if not isinstance(reply, dict):
            reply = {}
        tool_results = []
        for result in record.get("tool_results") or []:
            if not isinstance(result, dict):
                continue
            tool_results.append(
                {
                    "tool": result.get("tool"),
                    "ok": result.get("ok"),
                    "error": result.get("error"),
                }
            )
        compact.append(
            {
                "step": record.get("step"),
                "kind": record.get("kind"),
                "reason": reply.get("reason") or record.get("model_error"),
                "actions": record.get("actions") or reply.get("actions") or [],
                "tool_results": tool_results,
                "screen_change": record.get("screen_change"),
                "observation_error": record.get("observation_error"),
            }
        )
    return compact


def render_memory_prompt() -> str:
    template = Template(MEMORY_PROMPT_PATH.read_text(encoding="utf-8"))
    return template.render()


def review_completed_run(
    *,
    task: str,
    trajectory: list[dict[str, Any]],
    final_reply: dict[str, Any],
    final_screenshot: dict[str, Any],
    model: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    """Use one model call to verify success and derive reusable lessons."""

    convo = Convo(system=render_memory_prompt())
    review_model = os.getenv("MMFCUA_REVIEW_MODEL", model or "gpt-5.4-mini")
    review_input = json.dumps(
        {
            "task": task,
            "final_agent_reply": final_reply,
            "trajectory": _review_trajectory(trajectory),
            "instruction": (
                "Verify completion and return only the shortest reusable playbook "
                "for the next similar task."
            ),
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    if final_screenshot.get("path"):
        convo.user(review_input, screenshot=final_screenshot)
    else:
        convo.user(review_input)

    attempts = [
        {
            "response_format": {"type": "json_object"},
            "reasoning_effort": os.getenv("MMFCUA_REASONING_EFFORT", "none"),
            "verbosity": os.getenv("MMFCUA_VERBOSITY", "low"),
            "max_completion_tokens": 1400,
        },
        {"response_format": {"type": "json_object"}},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            text = convo.complete(
                model=review_model,
                api_key=api_key,
                stream_output=os.getenv("MMFCUA_STREAM_MODEL_OUTPUT", "1").lower()
                not in {"0", "false", "no", "off"},
                stream_prefix="[memory] model stream: ",
                **kwargs,
            )
            return normalize_review(_extract_json(text))
        except Exception as error:
            last_error = error
            error_text = str(error).lower()
            if not any(
                name in error_text
                for name in (
                    "response_format",
                    "reasoning_effort",
                    "verbosity",
                    "max_completion_tokens",
                )
            ):
                raise

    if last_error:
        raise last_error
    raise RuntimeError("memory review did not return a response.")


def verification_passed(
    review: dict[str, Any],
    *,
    minimum_confidence: float = DEFAULT_VERIFICATION_CONFIDENCE,
) -> bool:
    verification = review.get("verification", {})
    return bool(
        verification.get("success")
        and _number(verification.get("confidence")) >= minimum_confidence
    )


def make_memory(
    *,
    recorder: RunRecorder,
    review: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "run_id": recorder.run_id,
        "task": recorder.task,
        "success": True,
        "started_at": recorder.started_at,
        "completed_at": utc_now(),
        "source_run": str(recorder.path),
        "coordinate_policy": (
            "Mouse coordinates are historical hints only. Locate semantic targets "
            "in the current screenshot and choose fresh coordinates."
        ),
        "verification": review["verification"],
        "playbook": review["playbook"],
    }


def save_memory(
    memory: dict[str, Any],
    *,
    directory: Path = MEMORIES_DIR,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    run_id = str(memory.get("run_id") or _new_run_id(str(memory.get("task", "task"))))
    path = directory / f"{run_id}.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
    return path
