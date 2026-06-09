"""Agent loop for primitive computer-use tasks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Template

from conversation import Convo
from tools import mouse_position, run_actions_safe, screenshot, tools_json_for_prompt


PROMPT_PATH = Path(__file__).resolve().with_name("prompt.j2")
PROMPT_CACHE_KEY = "mmfcua-computer-use-agent-v1"


@dataclass
class AgentLoopResult:
    done: bool
    steps: int
    final_reply: dict[str, Any]
    conversation: Convo


def log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def observe_screen() -> tuple[dict[str, Any], dict[str, Any], str | None]:
    error_parts = []
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
            "instruction": (
                "Analyze the new screenshot and continue. If screen_change.changed is false, "
                "do not assume the last action worked; choose a different method. Return only "
                "the next JSON response."
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
    }


def _completion_error_mentions(error: Exception, *needles: str) -> bool:
    message = str(error).lower()
    return any(needle.lower() in message for needle in needles)


def _complete_json(convo: Convo, *, model: str | None, api_key: str | None) -> dict[str, Any]:
    cache_kwargs = _cache_kwargs()
    attempts = [
        {"response_format": {"type": "json_object"}, **cache_kwargs},
        {"response_format": {"type": "json_object"}},
        cache_kwargs,
        {},
    ]
    last_error: Exception | None = None
    text = ""
    for kwargs in attempts:
        try:
            text = convo.complete(model=model, api_key=api_key, **kwargs)
            break
        except Exception as error:
            last_error = error
            if not _completion_error_mentions(error, "response_format", "prompt_cache", "cache_retention"):
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
    log("[observe] capturing initial screenshot and mouse position", enabled=verbose)
    first_shot, first_mouse, first_error = observe_screen()
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

    for step in range(1, max_steps + 1):
        log(f"[step {step}] calling model={model!r}", enabled=verbose)
        try:
            reply = _complete_json(convo, model=model, api_key=api_key)
        except Exception as error:
            log(f"[step {step}] model/parse error: {error}", enabled=verbose)
            log(f"[step {step}] capturing recovery observation", enabled=verbose)
            shot, mouse, observe_error = observe_screen()
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
            log(f"[agent] task complete at step {step}", enabled=verbose)
            return AgentLoopResult(done=True, steps=step, final_reply=reply, conversation=convo)

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
        log(f"[conversation] post-action observation appended, messages={len(convo)}", enabled=verbose)
        previous_shot = shot

    log(f"[agent] stopped after max_steps={max_steps}, done=False", enabled=verbose)
    return AgentLoopResult(done=False, steps=max_steps, final_reply=last_reply, conversation=convo)
