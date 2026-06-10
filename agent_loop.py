"""Agent loop for primitive computer-use tasks."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Template

from conversation import Convo
from memory import (
    RunRecorder,
    make_memory,
    review_completed_run,
    save_memory,
    verification_passed,
)
from memory_search import MemoryVectorStore, guidance_message, memory_save_decision
from tools import mouse_position, run_actions_safe, screenshot, tools_json_for_prompt


PROMPT_PATH = Path(__file__).resolve().with_name("prompt.j2")
PROMPT_CACHE_KEY = "mmfcua-computer-use-agent-v1"


@dataclass
class AgentLoopResult:
    done: bool
    steps: int
    final_reply: dict[str, Any]
    conversation: Convo
    run_path: str | None = None
    memory_path: str | None = None
    memory_verified: bool | None = None
    memory_review: dict[str, Any] | None = None
    memory_error: str | None = None


def log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def _record_step(
    recorder: RunRecorder | None,
    record: dict[str, Any],
    *,
    verbose: bool,
) -> str | None:
    if recorder is None:
        return None
    try:
        recorder.record_step(record)
    except Exception as error:
        log(f"[memory] failed to record step: {error}", enabled=verbose)
        return str(error)
    return None


def _finish_run(
    recorder: RunRecorder | None,
    status: str,
    *,
    verbose: bool,
    **details: Any,
) -> None:
    if recorder is None:
        return
    try:
        recorder.finish(status, **details)
    except Exception as error:
        log(f"[memory] failed to finish run log: {error}", enabled=verbose)


def _minimum_memory_confidence() -> float:
    raw_value = os.getenv("MMFCUA_MEMORY_MIN_CONFIDENCE", "0.75")
    try:
        return max(0.0, min(1.0, float(raw_value)))
    except ValueError:
        return 0.75


def _observation_settle_seconds() -> float:
    raw_value = os.getenv("MMFCUA_OBSERVATION_SETTLE_SECONDS", "0.1")
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 0.1


def observe_screen() -> tuple[dict[str, Any], dict[str, Any], str | None]:
    error_parts = []
    time.sleep(_observation_settle_seconds())
    try:
        shot = screenshot()
    except Exception as error:
        shot = {"path": None, "width": None, "height": None, "error": str(error)}
        error_parts.append(f"screenshot failed: {error}")

    try:
        mouse = mouse_position()
    except Exception as error:
        mouse = {"x": None, "y": None, "error": str(error)}
        error_parts.append(f"mouse position failed: {error}")

    return shot, mouse, "; ".join(error_parts) or None


def screenshot_change(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_path = previous.get("path")
    current_path = current.get("path")
    if not previous_path or not current_path:
        return {"changed": None, "mean_delta": None, "reason": "missing screenshot path"}

    try:
        from PIL import Image, ImageChops

        with Image.open(previous_path) as before, Image.open(current_path) as after:
            before = before.convert("RGB").resize((320, 200))
            after = after.convert("RGB").resize((320, 200))
            diff = ImageChops.difference(before, after)
            histogram = diff.histogram()
            total = sum(value * (index % 256) for index, value in enumerate(histogram))
            pixels = before.size[0] * before.size[1] * 3
            mean_delta = total / pixels
            return {
                "changed": mean_delta > 1.5,
                "mean_delta": round(mean_delta, 3),
                "reason": "image diff",
            }
    except Exception as error:
        return {"changed": None, "mean_delta": None, "reason": str(error)}


def render_system_prompt(task: str, shot: dict[str, Any], mouse: dict[str, Any]) -> str:
    template = Template(PROMPT_PATH.read_text(encoding="utf-8"))
    return template.render(
        task=task,
        screenshot="Attached as an image in the user message, not embedded in this prompt.",
        screenshot_width=shot.get("width"),
        screenshot_height=shot.get("height"),
        mouse_x=mouse.get("x"),
        mouse_y=mouse.get("y"),
        tools_json=tools_json_for_prompt(),
    )


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
        raise TypeError("agent response must be a JSON object.")
    return data


def _normalize_reply(reply: dict[str, Any]) -> dict[str, Any]:
    done = bool(reply.get("done", False))
    actions = reply.get("actions", [])
    if actions is None:
        actions = []
    if not isinstance(actions, list):
        actions = []
    return {
        "reason": str(reply.get("reason", "")),
        "actions": actions,
        "done": done,
    }


def _observe_text(
    *,
    task: str,
    step: int,
    previous_reply: dict[str, Any],
    tool_results: list[dict[str, Any]],
    shot: dict[str, Any],
    mouse: dict[str, Any],
    screen_change: dict[str, Any],
) -> str:
    changed = screen_change.get("changed")
    if changed is False:
        action_feedback = (
            "The previous action batch had no visible effect. Do not repeat the "
            "same tools, nearby click coordinates, or the same strategy. Inspect "
            "whether a modal or foreground window blocks the target. If so, resolve "
            "only that blocker using a visible control, then observe again."
        )
    else:
        action_feedback = (
            "Use the new screenshot as the source of truth before selecting the "
            "next action."
        )

    return json.dumps(
        {
            "task": task,
            "step": step,
            "previous_reply": previous_reply,
            "tool_results": tool_results,
            "screenshot": {
                "width": shot.get("width"),
                "height": shot.get("height"),
            },
            "screen_change": screen_change,
            "mouse": mouse,
            "last_action_outcome": action_feedback,
            "instruction": (
                "Analyze the new screenshot and continue. Visible background targets "
                "are not actionable through a modal dialog. Resolve a blocker alone, "
                "observe the result, and only then interact with the target. Return "
                "only the next JSON response."
            ),
        },
        indent=2,
    )


def _cache_kwargs() -> dict[str, Any]:
    cache_key = os.getenv("LITELLM_PROMPT_CACHE_KEY", PROMPT_CACHE_KEY)
    cache_retention = os.getenv("LITELLM_PROMPT_CACHE_RETENTION", "in_memory")
    return {
        "prompt_cache_key": cache_key,
        "prompt_cache_retention": cache_retention,
    }


def _usage_summary(convo: Convo) -> dict[str, Any]:
    usage = convo.last_usage_dict()
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "time_to_first_token": convo.last_timing.get("time_to_first_token"),
        "total_seconds": convo.last_timing.get("total_seconds"),
    }


def _completion_error_mentions(error: Exception, *needles: str) -> bool:
    message = str(error).lower()
    return any(needle.lower() in message for needle in needles)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _is_openai_model(model: str | None) -> bool:
    if not model:
        return True
    return "/" not in model or model.startswith("openai/")


def _model_speed_kwargs(model: str | None) -> dict[str, Any]:
    if not _is_openai_model(model):
        return {}

    kwargs: dict[str, Any] = {
        "reasoning_effort": os.getenv("MMFCUA_REASONING_EFFORT", "none"),
        "verbosity": os.getenv("MMFCUA_VERBOSITY", "low"),
    }
    max_tokens = os.getenv("MMFCUA_MAX_COMPLETION_TOKENS", "800")
    try:
        kwargs["max_completion_tokens"] = max(1, int(max_tokens))
    except ValueError:
        kwargs["max_completion_tokens"] = 800
    service_tier = os.getenv("OPENAI_SERVICE_TIER")
    if service_tier:
        kwargs["service_tier"] = service_tier
    return kwargs


def _complete_json(
    convo: Convo,
    *,
    model: str | None,
    api_key: str | None,
    stream_prefix: str,
    memory_guidance: str | None = None,
) -> dict[str, Any]:
    cache_kwargs = _cache_kwargs()
    speed_kwargs = _model_speed_kwargs(model)
    attempts = [
        {"response_format": {"type": "json_object"}, **cache_kwargs, **speed_kwargs},
        {"response_format": {"type": "json_object"}, **speed_kwargs},
        {**cache_kwargs, **speed_kwargs},
        speed_kwargs,
        {},
    ]
    last_error: Exception | None = None
    text = ""
    for kwargs in attempts:
        try:
            text = convo.complete(
                model=model,
                api_key=api_key,
                stream_output=_env_bool("MMFCUA_STREAM_MODEL_OUTPUT", True),
                stream_prefix=stream_prefix,
                extra_messages=(
                    [{"role": "user", "content": memory_guidance}]
                    if memory_guidance
                    else None
                ),
                **kwargs,
            )
            break
        except Exception as error:
            last_error = error
            if not _completion_error_mentions(
                error,
                "response_format",
                "prompt_cache",
                "cache_retention",
                "reasoning_effort",
                "verbosity",
                "service_tier",
            ):
                raise
    else:
        if last_error:
            raise last_error
    return _normalize_reply(_extract_json(text))


def run_agent_loop(
    task: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_steps: int = 12,
    verbose: bool = True,
) -> AgentLoopResult:
    log(f"[agent] starting task: {task!r}", enabled=verbose)
    recorder: RunRecorder | None = None
    recorder_error: str | None = None
    try:
        recorder = RunRecorder(task)
        log(f"[memory] incremental run log={recorder.path}", enabled=verbose)
    except Exception as error:
        recorder_error = str(error)
        log(f"[memory] could not start run recorder: {error}", enabled=verbose)

    log("[observe] capturing initial screenshot and mouse position", enabled=verbose)
    first_shot, first_mouse, first_error = observe_screen()
    if recorder is not None:
        try:
            recorder.record_initial_observation(first_shot, first_mouse, first_error)
        except Exception as error:
            recorder_error = str(error)
            log(f"[memory] failed to record initial observation: {error}", enabled=verbose)
    if first_error:
        log(f"[observe] initial observation error: {first_error}", enabled=verbose)
    log(
        "[observe] initial screenshot="
        f"{first_shot.get('path')} size={first_shot.get('width')}x{first_shot.get('height')} "
        f"mouse=({first_mouse.get('x')}, {first_mouse.get('y')})",
        enabled=verbose,
    )

    convo = Convo(system=render_system_prompt(task, first_shot, first_mouse))
    log(f"[prompt] system prompt rendered, messages={len(convo)}", enabled=verbose)
    first_text = json.dumps(
        {
            "task": task,
            "observation_error": first_error,
            "instruction": "Use the attached screenshot when available. Return only JSON with done/actions/reason.",
        },
        indent=2,
    )
    if first_shot.get("path"):
        convo.user(first_text, screenshot=first_shot)
    else:
        convo.user(first_text)
    log(f"[conversation] initial user observation appended, messages={len(convo)}", enabled=verbose)

    last_reply: dict[str, Any] = {"reason": "", "actions": [], "done": False}
    previous_shot = first_shot
    memory_store: MemoryVectorStore | None = None
    memory_retrieval_error: str | None = None
    try:
        memory_store = MemoryVectorStore(api_key=api_key)
    except Exception as error:
        memory_retrieval_error = str(error)
        log(f"[memory-search] unavailable: {error}", enabled=verbose)

    for step in range(1, max_steps + 1):
        retrieved_memories: list[dict[str, Any]] = []
        retrieved_guidance: str | None = None
        if memory_store is not None:
            try:
                retrieved_memories = memory_store.search(
                    task,
                    limit=int(os.getenv("MMFCUA_MEMORY_SEARCH_LIMIT", "2")),
                    minimum_score=float(
                        os.getenv("MMFCUA_MEMORY_SEARCH_MIN_SCORE", "0.72")
                    ),
                )
                retrieved_guidance = guidance_message(retrieved_memories)
                if retrieved_memories:
                    scores = [match["score"] for match in retrieved_memories]
                    match_types = [
                        match.get("match_type")
                        for match in retrieved_memories
                    ]
                    log(
                        f"[memory-search] step={step} hits={len(scores)} "
                        f"scores={scores} types={match_types}",
                        enabled=verbose,
                    )
                else:
                    log(f"[memory-search] step={step} no relevant memory", enabled=verbose)
            except Exception as error:
                memory_retrieval_error = str(error)
                log(f"[memory-search] step={step} failed: {error}", enabled=verbose)

        log(f"[step {step}] calling model={model!r}", enabled=verbose)
        try:
            reply = _complete_json(
                convo,
                model=model,
                api_key=api_key,
                stream_prefix=f"[step {step}] model stream: ",
                memory_guidance=retrieved_guidance,
            )
        except Exception as error:
            log(f"[step {step}] model/parse error: {error}", enabled=verbose)
            log(f"[step {step}] capturing recovery observation", enabled=verbose)
            shot, mouse, observe_error = observe_screen()
            change = screenshot_change(previous_shot, shot)
            text = json.dumps(
                {
                    "model_response_error": str(error),
                    "observation_error": observe_error,
                    "instruction": "Your previous response was not usable. Return valid JSON only.",
                },
                indent=2,
            )
            if shot.get("path"):
                convo.observe(text, screenshot=shot)
            else:
                convo.observe(text)
            last_reply = {"reason": "Invalid model response.", "actions": [], "done": False}
            record_error = _record_step(
                recorder,
                {
                    "step": step,
                    "kind": "model_error",
                    "model_error": str(error),
                    "screen_before": previous_shot,
                    "screen_after": shot,
                    "screen_change": change,
                    "mouse_after": mouse,
                    "observation_error": observe_error,
                    "retrieved_memories": retrieved_memories,
                },
                verbose=verbose,
            )
            recorder_error = record_error or recorder_error
            previous_shot = shot
            log(f"[step {step}] recovery observation appended, messages={len(convo)}", enabled=verbose)
            continue

        last_reply = reply
        log(
            f"[step {step}] model reply done={reply['done']} "
            f"actions={len(reply['actions'])} reason={reply['reason']!r}",
            enabled=verbose,
        )
        log(f"[step {step}] usage/cache: {json.dumps(_usage_summary(convo))}", enabled=verbose)
        log(f"[step {step}] parsed reply: {json.dumps(reply, ensure_ascii=False)}", enabled=verbose)
        if reply["done"]:
            record_error = _record_step(
                recorder,
                {
                    "step": step,
                    "kind": "completion_claim",
                    "reply": reply,
                    "screen_before": previous_shot,
                    "screen_after": previous_shot,
                    "screen_change": {
                        "changed": None,
                        "mean_delta": None,
                        "reason": "no action; agent claimed completion",
                    },
                    "retrieved_memories": retrieved_memories,
                },
                verbose=verbose,
            )
            recorder_error = record_error or recorder_error

            memory_path: str | None = None
            memory_verified: bool | None = None
            memory_review: dict[str, Any] | None = None
            memory_error = recorder_error or memory_retrieval_error

            if recorder is not None:
                log("[memory] verifying and reflecting on completed run", enabled=verbose)
                try:
                    memory_review = review_completed_run(
                        task=task,
                        trajectory=recorder.trajectory,
                        final_reply=reply,
                        final_screenshot=previous_shot,
                        model=model,
                        api_key=api_key,
                    )
                    memory_verified = verification_passed(
                        memory_review,
                        minimum_confidence=_minimum_memory_confidence(),
                    )
                    verification = memory_review["verification"]
                    log(
                        "[memory] verification "
                        f"success={verification['success']} "
                        f"confidence={verification['confidence']}",
                        enabled=verbose,
                    )
                    if memory_verified:
                        candidate_memory = make_memory(
                            recorder=recorder,
                            review=memory_review,
                        )
                        should_save, save_reason = memory_save_decision(
                            candidate_memory,
                            retrieved_memories,
                        )
                        if should_save:
                            saved_path = save_memory(candidate_memory)
                            memory_path = str(saved_path)
                            log(
                                f"[memory] saved reusable memory={memory_path} "
                                f"reason={save_reason}",
                                enabled=verbose,
                            )
                            if memory_store is not None:
                                try:
                                    memory_store.refresh()
                                    log("[memory-search] indexed new memory", enabled=verbose)
                                except Exception as error:
                                    memory_retrieval_error = str(error)
                                    memory_error = str(error)
                                    log(
                                        f"[memory-search] could not index new memory: {error}",
                                        enabled=verbose,
                                    )
                        else:
                            memory_path = next(
                                (
                                    match.get("path")
                                    for match in retrieved_memories
                                    if match.get("match_type") == "exact_task"
                                ),
                                None,
                            )
                            log(
                                f"[memory] reused existing memory path={memory_path} "
                                f"reason={save_reason}",
                                enabled=verbose,
                            )
                    else:
                        log("[memory] verification rejected; reusable memory not saved", enabled=verbose)
                except Exception as error:
                    memory_error = str(error)
                    log(f"[memory] post-task review failed: {error}", enabled=verbose)

            if memory_path and any(
                match.get("path") == memory_path
                for match in retrieved_memories
            ):
                finish_status = "memory_reused"
            else:
                finish_status = "memory_saved" if memory_path else "completed_without_memory"
            _finish_run(
                recorder,
                finish_status,
                verbose=verbose,
                agent_done=True,
                memory_verified=memory_verified,
                memory_path=memory_path,
                memory_error=memory_error,
            )
            log(f"[agent] task complete at step {step}", enabled=verbose)
            return AgentLoopResult(
                done=True,
                steps=step,
                final_reply=reply,
                conversation=convo,
                run_path=str(recorder.path) if recorder else None,
                memory_path=memory_path,
                memory_verified=memory_verified,
                memory_review=memory_review,
                memory_error=memory_error,
            )

        log(f"[step {step}] executing actions: {json.dumps(reply['actions'], ensure_ascii=False)}", enabled=verbose)
        tool_results = run_actions_safe(reply["actions"])
        log(f"[step {step}] tool results: {json.dumps(tool_results, ensure_ascii=False)}", enabled=verbose)
        log(f"[step {step}] capturing post-action screenshot and mouse position", enabled=verbose)
        shot, mouse, observe_error = observe_screen()
        if observe_error:
            log(f"[step {step}] observation error: {observe_error}", enabled=verbose)
        change = screenshot_change(previous_shot, shot)
        log(f"[observe] screen change: {json.dumps(change)}", enabled=verbose)
        log(
            "[observe] post-action screenshot="
            f"{shot.get('path')} size={shot.get('width')}x{shot.get('height')} "
            f"mouse=({mouse.get('x')}, {mouse.get('y')})",
            enabled=verbose,
        )
        observation_text = _observe_text(
            task=task,
            step=step,
            previous_reply=reply,
            tool_results=tool_results,
            shot=shot,
            mouse=mouse,
            screen_change=change,
        )
        if observe_error:
            observation_text = json.dumps(
                {
                    "observation_error": observe_error,
                    "observation": json.loads(observation_text),
                },
                indent=2,
            )
        if shot.get("path"):
            convo.observe(observation_text, screenshot=shot)
        else:
            convo.observe(observation_text)
        record_error = _record_step(
            recorder,
            {
                "step": step,
                "kind": "action",
                "reply": reply,
                "actions": reply["actions"],
                "tool_results": tool_results,
                "screen_before": previous_shot,
                "screen_after": shot,
                "screen_change": change,
                "mouse_after": mouse,
                "observation_error": observe_error,
                "retrieved_memories": retrieved_memories,
                "coordinate_policy": (
                    "Coordinates are historical hints only; relocate semantic "
                    "targets from the current screenshot before reuse."
                ),
            },
            verbose=verbose,
        )
        recorder_error = record_error or recorder_error
        log(f"[conversation] post-action observation appended, messages={len(convo)}", enabled=verbose)
        previous_shot = shot

    log(f"[agent] stopped after max_steps={max_steps}, done=False", enabled=verbose)
    _finish_run(
        recorder,
        "max_steps_reached",
        verbose=verbose,
        agent_done=False,
        memory_error=recorder_error,
    )
    return AgentLoopResult(
        done=False,
        steps=max_steps,
        final_reply=last_reply,
        conversation=convo,
        run_path=str(recorder.path) if recorder else None,
        memory_error=recorder_error,
    )
