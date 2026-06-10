"""Small LiteLLM-ready conversation store for text + screenshots."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
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
        self.last_timing: dict[str, float | None] = {}
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
        stream_output: bool = False,
        stream_prefix: str = "[model] ",
        **kwargs: Any,
    ) -> str:
        from dotenv import load_dotenv
        from litellm import completion, stream_chunk_builder

        load_dotenv()
        messages = self.to_litellm()
        request_started = time.perf_counter()
        first_token_at: float | None = None

        if stream_output:
            response_stream = completion(
                model=model or os.getenv("LITELLM_MODEL", "gpt-5.4-mini"),
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                messages=messages,
                stream=True,
                **kwargs,
            )
            chunks = []
            text_parts = []
            print(stream_prefix, end="", flush=True)
            for chunk in response_stream:
                chunks.append(chunk)
                choices = getattr(chunk, "choices", None) or []
                delta = getattr(choices[0], "delta", None) if choices else None
                content = getattr(delta, "content", None)
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(content)
                    print(content, end="", flush=True)
            print(flush=True)
            text = "".join(text_parts)
            try:
                response = stream_chunk_builder(chunks, messages=messages)
            except Exception:
                response = chunks[-1] if chunks else None
        else:
            response = completion(
                model=model or os.getenv("LITELLM_MODEL", "gpt-5.4-mini"),
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                messages=messages,
                **kwargs,
            )
            text = response.choices[0].message.content or ""
            if text:
                first_token_at = time.perf_counter()

        completed_at = time.perf_counter()
        self.last_response = response
        self.last_usage = getattr(response, "usage", None)
        self.last_timing = {
            "time_to_first_token": (
                round(first_token_at - request_started, 3)
                if first_token_at is not None
                else None
            ),
            "total_seconds": round(completed_at - request_started, 3),
        }
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

    def to_litellm(self, *, max_images: int = 1) -> list[Message]:
        """Return all text history while retaining only the newest screenshots."""

        remaining_images = max(0, max_images)
        prepared: list[Message] = []
        for message in reversed(self._messages):
            content = message.get("content")
            if not isinstance(content, list):
                prepared.append(dict(message))
                continue

            filtered_content = []
            for part in reversed(content):
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    filtered_content.append(part)
                    continue
                if remaining_images:
                    filtered_content.append(part)
                    remaining_images -= 1
            prepared.append({**message, "content": list(reversed(filtered_content))})

        return list(reversed(prepared))

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
