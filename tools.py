"""Primitive computer-use tools backed by pyautogui."""

from __future__ import annotations

import time
import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ToolFunction = Callable[..., Any]

_COORDINATE_SPACE: dict[str, int | None] = {
    "screenshot_width": None,
    "screenshot_height": None,
    "screen_width": None,
    "screen_height": None,
}


@dataclass(frozen=True)
class tool:
    name: str
    description: str
    arguments: dict[str, Any]
    function: ToolFunction

    def call(self, arguments: dict[str, Any] | None = None) -> Any:
        return self.function(**(arguments or {}))

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


def _pyautogui():
    import pyautogui

    pyautogui.PAUSE = 0
    return pyautogui


def _require_number(name: str, value: Any) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number.")
    return float(value)


def _require_button(button: str) -> str:
    if button not in {"left", "right", "middle"}:
        raise ValueError("button must be one of: left, right, middle.")
    return button


def _remember_coordinate_space(
    screenshot_width: int | None,
    screenshot_height: int | None,
    screen_width: int | None,
    screen_height: int | None,
) -> None:
    if not screenshot_width or not screenshot_height or not screen_width or not screen_height:
        return
    _COORDINATE_SPACE.update(
        {
            "screenshot_width": int(screenshot_width),
            "screenshot_height": int(screenshot_height),
            "screen_width": int(screen_width),
            "screen_height": int(screen_height),
        }
    )


def _screen_size_or_none() -> tuple[int | None, int | None]:
    try:
        size = _pyautogui().size()
    except Exception:
        return None, None
    if not size.width or not size.height:
        return None, None
    return int(size.width), int(size.height)


def _to_screen_coordinates(x: int | float, y: int | float) -> tuple[float, float]:
    x_pos = _require_number("x", x)
    y_pos = _require_number("y", y)

    screenshot_width = _COORDINATE_SPACE["screenshot_width"]
    screenshot_height = _COORDINATE_SPACE["screenshot_height"]
    screen_width = _COORDINATE_SPACE["screen_width"]
    screen_height = _COORDINATE_SPACE["screen_height"]

    if screenshot_width and screenshot_height and screen_width and screen_height:
        return (
            x_pos * (screen_width / screenshot_width),
            y_pos * (screen_height / screenshot_height),
        )

    return x_pos, y_pos


def _from_screen_coordinates(x: int | float, y: int | float) -> tuple[float, float]:
    x_pos = _require_number("x", x)
    y_pos = _require_number("y", y)

    screenshot_width = _COORDINATE_SPACE["screenshot_width"]
    screenshot_height = _COORDINATE_SPACE["screenshot_height"]
    screen_width = _COORDINATE_SPACE["screen_width"]
    screen_height = _COORDINATE_SPACE["screen_height"]

    if screenshot_width and screenshot_height and screen_width and screen_height:
        return (
            x_pos * (screenshot_width / screen_width),
            y_pos * (screenshot_height / screen_height),
        )

    return x_pos, y_pos


KEY_MAP = {
    "Command": "command",
    "Control": "ctrl",
    "Option": "option",
    "Shift": "shift",
    "Fn": "fn",
    "Enter": "enter",
    "Escape": "esc",
    "Tab": "tab",
    "Space": "space",
    "Backspace": "backspace",
    "Delete": "delete",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "Home": "home",
    "End": "end",
    "PageUp": "pageup",
    "PageDown": "pagedown",
}


def _normalize_key(key: str) -> str:
    if key in KEY_MAP:
        return KEY_MAP[key]
    if len(key) == 1 and (key.isupper() or key.isdigit()):
        return key.lower()
    if key.startswith("F") and key[1:].isdigit() and 1 <= int(key[1:]) <= 12:
        return key.lower()
    raise ValueError(f"Unsupported key name: {key!r}. Use the canonical names from prompt.j2.")


def mouse_move(x: int | float, y: int | float) -> dict[str, Any]:
    pg = _pyautogui()
    screen_x, screen_y = _to_screen_coordinates(x, y)
    pg.moveTo(screen_x, screen_y)
    return {"x": x, "y": y, "screen_x": screen_x, "screen_y": screen_y}


def mouse_click(button: str, x: int | float, y: int | float) -> dict[str, Any]:
    pg = _pyautogui()
    screen_x, screen_y = _to_screen_coordinates(x, y)
    pg.click(x=screen_x, y=screen_y, button=_require_button(button))
    return {"button": button, "x": x, "y": y, "screen_x": screen_x, "screen_y": screen_y}


def mouse_double_click(button: str, x: int | float, y: int | float) -> dict[str, Any]:
    pg = _pyautogui()
    screen_x, screen_y = _to_screen_coordinates(x, y)
    pg.doubleClick(x=screen_x, y=screen_y, button=_require_button(button))
    return {"button": button, "x": x, "y": y, "screen_x": screen_x, "screen_y": screen_y}


def mouse_down(button: str, x: int | float, y: int | float) -> dict[str, Any]:
    pg = _pyautogui()
    screen_x, screen_y = _to_screen_coordinates(x, y)
    pg.moveTo(screen_x, screen_y)
    pg.mouseDown(button=_require_button(button))
    return {"button": button, "x": x, "y": y, "screen_x": screen_x, "screen_y": screen_y}


def mouse_up(button: str, x: int | float, y: int | float) -> dict[str, Any]:
    pg = _pyautogui()
    screen_x, screen_y = _to_screen_coordinates(x, y)
    pg.moveTo(screen_x, screen_y)
    pg.mouseUp(button=_require_button(button))
    return {"button": button, "x": x, "y": y, "screen_x": screen_x, "screen_y": screen_y}


def mouse_drag(
    button: str,
    x: int | float,
    y: int | float,
    to_x: int | float,
    to_y: int | float,
    duration: int | float = 0.2,
) -> dict[str, Any]:
    pg = _pyautogui()
    screen_x, screen_y = _to_screen_coordinates(x, y)
    screen_to_x, screen_to_y = _to_screen_coordinates(to_x, to_y)
    drag_duration = _require_number("duration", duration)
    pg.moveTo(screen_x, screen_y)
    pg.dragTo(screen_to_x, screen_to_y, duration=drag_duration, button=_require_button(button))
    return {
        "button": button,
        "x": x,
        "y": y,
        "to_x": to_x,
        "to_y": to_y,
        "screen_x": screen_x,
        "screen_y": screen_y,
        "screen_to_x": screen_to_x,
        "screen_to_y": screen_to_y,
        "duration": drag_duration,
    }


def type_text(text: str, interval: int | float = 0) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    pg = _pyautogui()
    pg.write(text, interval=_require_number("interval", interval))
    return {"text": text}


def key_press(key: str) -> dict[str, Any]:
    pg = _pyautogui()
    normalized = _normalize_key(key)
    pg.press(normalized)
    return {"key": key}


def hotkey(keys: list[str]) -> dict[str, Any]:
    if not isinstance(keys, list) or not keys:
        raise TypeError("keys must be a non-empty list of canonical key names.")
    normalized = [_normalize_key(key) for key in keys]

    pg = _pyautogui()
    modifiers = normalized[:-1]
    final_key = normalized[-1]
    for key in modifiers:
        pg.keyDown(key)
        time.sleep(0.08)
    time.sleep(0.12)
    pg.press(final_key)
    time.sleep(0.12)
    for key in reversed(modifiers):
        pg.keyUp(key)
        time.sleep(0.08)
    time.sleep(0.7)
    return {"keys": keys, "normalized_keys": normalized}


def scroll(x: int | float = 0, y: int | float = 0) -> dict[str, Any]:
    pg = _pyautogui()
    x_amount = int(_require_number("x", x))
    y_amount = int(_require_number("y", y))
    if y_amount:
        pg.scroll(y_amount)
    if x_amount:
        pg.hscroll(x_amount)
    return {"x": x_amount, "y": y_amount}


def wait(seconds: int | float) -> dict[str, Any]:
    delay = _require_number("seconds", seconds)
    if delay < 0:
        raise ValueError("seconds must be non-negative.")
    time.sleep(delay)
    return {"seconds": delay}


def _screenshot_path(path: str | None = None) -> Path:
    output_path = Path(path) if path else Path(".screenshots") / f"screenshot_{time.time_ns()}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as file:
        header = file.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", header[16:24])
        return width, height
    return None, None


def screenshot(path: str | None = None) -> dict[str, Any]:
    output_path = _screenshot_path(path)
    screen_width, screen_height = _screen_size_or_none()

    try:
        pg = _pyautogui()
        image = pg.screenshot()
        image.save(output_path)
        _remember_coordinate_space(image.width, image.height, screen_width, screen_height)
        return {
            "path": str(output_path),
            "width": image.width,
            "height": image.height,
            "screen_width": screen_width,
            "screen_height": screen_height,
            "method": "pyautogui",
        }
    except Exception:
        subprocess.run(
            ["/usr/sbin/screencapture", "-x", "-t", "png", str(output_path)],
            check=True,
        )

    width, height = _png_size(output_path)
    _remember_coordinate_space(width, height, screen_width, screen_height)
    return {
        "path": str(output_path),
        "width": width,
        "height": height,
        "screen_width": screen_width,
        "screen_height": screen_height,
        "method": "screencapture",
    }


def mouse_position() -> dict[str, Any]:
    pg = _pyautogui()
    position = pg.position()
    x, y = _from_screen_coordinates(position.x, position.y)
    return {"x": x, "y": y, "screen_x": position.x, "screen_y": position.y}


def screen_size() -> dict[str, Any]:
    pg = _pyautogui()
    size = pg.size()
    return {"width": size.width, "height": size.height}


TOOLS = [
    tool(
        name="mouse_move",
        description="Move the mouse pointer to absolute screen coordinates.",
        arguments={"x": "number", "y": "number"},
        function=mouse_move,
    ),
    tool(
        name="mouse_click",
        description="Click one mouse button at absolute screen coordinates.",
        arguments={"button": "left | right | middle", "x": "number", "y": "number"},
        function=mouse_click,
    ),
    tool(
        name="mouse_double_click",
        description="Double-click one mouse button at absolute screen coordinates.",
        arguments={"button": "left | right | middle", "x": "number", "y": "number"},
        function=mouse_double_click,
    ),
    tool(
        name="mouse_down",
        description="Move to absolute screen coordinates and hold one mouse button down.",
        arguments={"button": "left | right | middle", "x": "number", "y": "number"},
        function=mouse_down,
    ),
    tool(
        name="mouse_up",
        description="Move to absolute screen coordinates and release one mouse button.",
        arguments={"button": "left | right | middle", "x": "number", "y": "number"},
        function=mouse_up,
    ),
    tool(
        name="mouse_drag",
        description="Drag from one absolute coordinate to another.",
        arguments={
            "button": "left | right | middle",
            "x": "number",
            "y": "number",
            "to_x": "number",
            "to_y": "number",
            "duration": "number, optional seconds",
        },
        function=mouse_drag,
    ),
    tool(
        name="type_text",
        description="Type literal text into the focused UI element.",
        arguments={"text": "string", "interval": "number, optional seconds between characters"},
        function=type_text,
    ),
    tool(
        name="key_press",
        description="Press one canonical keyboard key.",
        arguments={"key": "canonical key name from prompt.j2"},
        function=key_press,
    ),
    tool(
        name="hotkey",
        description="Press a keyboard shortcut. Use canonical keys, modifiers first.",
        arguments={"keys": "list of canonical key names, e.g. ['Command', 'A']"},
        function=hotkey,
    ),
    tool(
        name="scroll",
        description="Scroll vertically and/or horizontally.",
        arguments={"x": "integer horizontal amount", "y": "integer vertical amount"},
        function=scroll,
    ),
    tool(
        name="wait",
        description="Wait for UI changes or animations.",
        arguments={"seconds": "number"},
        function=wait,
    ),
    tool(
        name="screenshot",
        description="Capture the current screen and return the saved image path and size.",
        arguments={"path": "string, optional output PNG path"},
        function=screenshot,
    ),
    tool(
        name="mouse_position",
        description="Return the current mouse pointer coordinates.",
        arguments={},
        function=mouse_position,
    ),
    tool(
        name="screen_size",
        description="Return the current screen width and height.",
        arguments={},
        function=screen_size,
    ),
]

TOOLS_BY_NAME = {entry.name: entry for entry in TOOLS}


def tools_json() -> list[dict[str, Any]]:
    return [entry.to_json() for entry in TOOLS]


def tools_json_for_prompt(indent: int = 2) -> str:
    return json.dumps(tools_json(), indent=indent)


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    if name not in TOOLS_BY_NAME:
        raise KeyError(f"Unknown tool: {name}")
    return TOOLS_BY_NAME[name].call(arguments)


def run_actions(actions: list[dict[str, Any]]) -> list[Any]:
    results = []
    for action in actions:
        results.append(call_tool(action["tool"], action.get("arguments", {})))
    return results


def run_actions_safe(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for index, action in enumerate(actions):
        try:
            if not isinstance(action, dict):
                raise TypeError("action must be an object.")
            name = action.get("tool")
            arguments = action.get("arguments", {})
            if not isinstance(name, str):
                raise TypeError("action.tool must be a string.")
            if not isinstance(arguments, dict):
                raise TypeError("action.arguments must be an object.")
            result = call_tool(name, arguments)
            results.append({"index": index, "tool": name, "ok": True, "result": result})
        except Exception as error:
            results.append({
                "index": index,
                "tool": action.get("tool") if isinstance(action, dict) else None,
                "ok": False,
                "error": str(error),
            })
    return results
