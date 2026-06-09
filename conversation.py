"""Small LiteLLM-ready conversation store for text + screenshots."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterator


Message = dict[str, Any]
Screenshot = str | Path | dict[str, Any]


def image_data_uri(image: str | Path) -> str:
    path = Path(image)
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _screenshot_url(screenshot: Screenshot) -> str:
    if isinstance(screenshot, dict):
        if "data_uri" in screenshot:
            return str(screenshot["data_uri"])
        if "path" in screenshot:
            return image_data_uri(screenshot["path"])
        raise ValueError("screenshot dict must include 'path' or 'data_uri'.")

    screenshot_url = str(screenshot)
    if screenshot_url.startswith("data:"):
        return screenshot_url
    return image_data_uri(screenshot)


def user_content(text: str, screenshot: Screenshot | None = None) -> str | list[dict[str, Any]]:
    if screenshot is None:
        return text

    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": _screenshot_url(screenshot)}},
    ]


class Conversation:
    """A tiny wrapper around LiteLLM messages.

    The trick is that this class behaves like a message list for LiteLLM, but
    gives names to the operations we care about: user text+screenshot in,
    assistant JSON/text out.
    """

    def __init__(self, system: str | None = None) -> None:
        self._messages: list[Message] = []
        self.last_response: Any | None = None
        self.last_usage: Any | None = None
        if system:
            self.system(system)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: int) -> Message:
        return self._messages[index]

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def system(self, text: str) -> Message:
        return self._append("system", text)

    def user(self, text: str, screenshot: Screenshot | None = None) -> Message:
        return self._append("user", user_content(text, screenshot))

    def observe(self, text: str, screenshot: Screenshot | None = None) -> Message:
        return self.user(text, screenshot=screenshot)

    def assistant(self, text: str) -> Message:
        return self._append("assistant", text)

    def _append(self, role: str, content: Any) -> Message:
        message = {"role": role, "content": content}
        self._messages.append(message)
        return message

    def complete(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> str:
        from dotenv import load_dotenv
        from litellm import completion

        load_dotenv()
        response = completion(
            model=model or os.getenv("LITELLM_MODEL", "gpt-4.1-nano"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            messages=self.to_litellm(),
            **kwargs,
        )
        self.last_response = response
        self.last_usage = getattr(response, "usage", None)
        text = response.choices[0].message.content or ""
        self.assistant(text)
        return text

    def ask(
        self,
        text: str,
        screenshot: Screenshot | None = None,
        **kwargs: Any,
    ) -> str:
        self.user(text, screenshot=screenshot)
        return self.complete(**kwargs)

    def to_litellm(self) -> list[Message]:
        return self._messages

    def copy_messages(self) -> list[Message]:
        return list(self._messages)

    def last_usage_dict(self) -> dict[str, Any]:
        usage = self.last_usage
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "dict"):
            return usage.dict()
        try:
            return json.loads(json.dumps(usage, default=lambda value: getattr(value, "__dict__", str(value))))
        except Exception:
            return {"raw": str(usage)}


Convo = Conversation
