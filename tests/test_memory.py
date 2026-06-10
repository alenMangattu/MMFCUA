from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent_loop
from conversation import Convo
from memory import (
    RunRecorder,
    make_memory,
    normalize_review,
    save_memory,
    verification_passed,
)
from tools import tools_json_for_prompt


def successful_review() -> dict:
    return {
        "verification": {
            "success": True,
            "confidence": 0.92,
            "summary": "The requested application is visibly open.",
        },
        "playbook": {
            "task_signature": "Open an installed application",
            "applicability": "Use when the requested app has a known installed alias.",
            "learned_target_mappings": [
                {
                    "requested": "Chrome",
                    "effective": "Chromium",
                    "relationship": "installed substitute",
                    "confidence": 0.9,
                    "revalidate_next_time": True,
                }
            ],
            "preferred_plan": [
                {
                    "intent": "Open the application",
                    "target": "Spotlight result for Chromium",
                    "method": "Search for Chromium and open the verified app result.",
                    "preconditions": ["Spotlight is visible."],
                    "success_check": "A Chromium browser window is visibly active.",
                }
            ],
            "fallbacks": [
                {
                    "when": "Spotlight does not launch the app",
                    "method": "Open Chromium from Finder Applications.",
                    "success_check": "A Chromium browser window is visibly active.",
                }
            ],
            "avoid": ["Do not replay historical mouse coordinates."],
            "timing_notes": ["Wait for the application window before typing again."],
            "environment_facts": [],
        },
    }


class MemoryTests(unittest.TestCase):
    def test_conversation_streams_model_output_to_terminal(self) -> None:
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content='{"done":'))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="true}"))]
            ),
        ]
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = lambda **kwargs: iter(chunks)
        fake_litellm.stream_chunk_builder = lambda chunks, messages: SimpleNamespace(
            usage={"prompt_tokens": 10}
        )

        convo = Convo(system="system")
        convo.user("task")
        output = io.StringIO()
        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            with redirect_stdout(output):
                text = convo.complete(
                    model="test-model",
                    api_key="test-key",
                    stream_output=True,
                    stream_prefix="[stream] ",
                )

        self.assertEqual(text, '{"done":true}')
        self.assertEqual(output.getvalue(), '[stream] {"done":true}\n')
        self.assertIsNotNone(convo.last_timing["time_to_first_token"])
        self.assertEqual(convo.messages[-1]["content"], '{"done":true}')

    def test_conversation_keeps_only_latest_screenshot_for_requests(self) -> None:
        convo = Convo(system="system")
        convo.user("first", screenshot={"data_uri": "data:image/png;base64,first"})
        convo.assistant("action")
        convo.user("second", screenshot={"data_uri": "data:image/png;base64,second"})

        messages = convo.to_litellm()
        image_urls = [
            part["image_url"]["url"]
            for message in messages
            if isinstance(message["content"], list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]

        self.assertEqual(image_urls, ["data:image/png;base64,second"])
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "first"}])

    def test_run_recorder_writes_incremental_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = RunRecorder("Open Chromium", directory=Path(temporary))
            recorder.record_initial_observation(
                {"path": ".screenshots/initial.png", "width": 100, "height": 100},
                {"x": 10, "y": 20},
                None,
            )
            recorder.record_step(
                {
                    "step": 1,
                    "kind": "action",
                    "actions": [{"tool": "mouse_click", "arguments": {"x": 10, "y": 20}}],
                }
            )
            recorder.finish("completed")

            events = [
                json.loads(line)
                for line in recorder.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                ["run_started", "initial_observation", "step", "run_finished"],
            )
            self.assertEqual(events[2]["actions"][0]["arguments"]["x"], 10)

    def test_normalization_removes_coordinate_fields_from_strategy(self) -> None:
        review = successful_review()
        strategy = review["playbook"]["preferred_plan"][0]
        strategy["x"] = 900
        strategy["coordinates"] = {"x": 900, "y": 400}

        normalized = normalize_review(review)
        normalized_strategy = normalized["playbook"]["preferred_plan"][0]

        self.assertNotIn("x", normalized_strategy)
        self.assertNotIn("coordinates", normalized_strategy)

    def test_memory_is_saved_atomically_after_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = RunRecorder("Open Chromium", directory=root / "runs")
            recorder.record_step({"step": 1, "kind": "completion_claim"})
            review = successful_review()

            self.assertTrue(verification_passed(review))
            memory = make_memory(
                recorder=recorder,
                review=review,
            )
            path = save_memory(memory, directory=root / "memories")
            stored = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(stored["task"], "Open Chromium")
            self.assertEqual(stored["verification"]["confidence"], 0.92)
            self.assertNotIn("trajectory", stored)
            self.assertNotIn("step_assessments", stored)
            self.assertEqual(
                stored["playbook"]["learned_target_mappings"][0]["effective"],
                "Chromium",
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_observation_waits_before_screenshot(self) -> None:
        with (
            patch("agent_loop.time.sleep") as sleep_mock,
            patch(
                "agent_loop.screenshot",
                return_value={"path": None, "width": 100, "height": 100},
            ),
            patch("agent_loop.mouse_position", return_value={"x": 1, "y": 2}),
        ):
            agent_loop.observe_screen()

        sleep_mock.assert_called_once_with(0.1)

    def test_automatic_observation_tools_are_hidden_from_agent_prompt(self) -> None:
        prompt_tools = json.loads(tools_json_for_prompt())
        names = {tool["name"] for tool in prompt_tools}

        self.assertNotIn("screenshot", names)
        self.assertNotIn("mouse_position", names)
        self.assertNotIn("screen_size", names)

    def test_agent_loop_records_steps_and_saves_verified_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations = iter(
                [
                    ({"path": None, "width": 100, "height": 100}, {"x": 1, "y": 2}, None),
                    ({"path": None, "width": 100, "height": 100}, {"x": 3, "y": 4}, None),
                ]
            )
            replies = iter(
                [
                    {
                        "reason": "Open the app.",
                        "actions": [{"tool": "key_press", "arguments": {"key": "Enter"}}],
                        "done": False,
                    },
                    {
                        "reason": "The app is visibly open.",
                        "actions": [],
                        "done": True,
                    },
                ]
            )

            def recorder_factory(task: str) -> RunRecorder:
                return RunRecorder(task, directory=root / "runs")

            def save_to_test_directory(memory: dict) -> Path:
                return save_memory(memory, directory=root / "memories")

            with (
                patch("agent_loop.RunRecorder", side_effect=recorder_factory),
                patch("agent_loop.observe_screen", side_effect=lambda: next(observations)),
                patch("agent_loop._complete_json", side_effect=lambda *args, **kwargs: next(replies)),
                patch(
                    "agent_loop.run_actions_safe",
                    return_value=[{"index": 0, "tool": "key_press", "ok": True, "result": {}}],
                ),
                patch("agent_loop.review_completed_run", return_value=successful_review()),
                patch("agent_loop.save_memory", side_effect=save_to_test_directory),
            ):
                result = agent_loop.run_agent_loop(
                    "Open Chromium",
                    model="test-model",
                    api_key="test-key",
                    max_steps=3,
                    verbose=False,
                )

            self.assertTrue(result.done)
            self.assertTrue(result.memory_verified)
            self.assertIsNotNone(result.memory_path)
            self.assertTrue(Path(result.memory_path).exists())

            run_events = [
                json.loads(line)
                for line in Path(result.run_path).read_text(encoding="utf-8").splitlines()
            ]
            step_events = [event for event in run_events if event["event"] == "step"]
            self.assertEqual([event["kind"] for event in step_events], ["action", "completion_claim"])
            self.assertEqual(run_events[-1]["status"], "memory_saved")

    def test_agent_loop_does_not_save_rejected_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rejected_review = successful_review()
            rejected_review["verification"] = {
                "success": False,
                "confidence": 0.35,
                "summary": "The requested result is not visible.",
            }

            def recorder_factory(task: str) -> RunRecorder:
                return RunRecorder(task, directory=root / "runs")

            with (
                patch("agent_loop.RunRecorder", side_effect=recorder_factory),
                patch(
                    "agent_loop.observe_screen",
                    return_value=(
                        {"path": None, "width": 100, "height": 100},
                        {"x": 1, "y": 2},
                        None,
                    ),
                ),
                patch(
                    "agent_loop._complete_json",
                    return_value={
                        "reason": "I think the task is complete.",
                        "actions": [],
                        "done": True,
                    },
                ),
                patch("agent_loop.review_completed_run", return_value=rejected_review),
                patch("agent_loop.save_memory") as save_mock,
            ):
                result = agent_loop.run_agent_loop(
                    "Open Chromium",
                    model="test-model",
                    api_key="test-key",
                    max_steps=1,
                    verbose=False,
                )

            self.assertTrue(result.done)
            self.assertFalse(result.memory_verified)
            self.assertIsNone(result.memory_path)
            save_mock.assert_not_called()

            run_events = [
                json.loads(line)
                for line in Path(result.run_path).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(run_events[-1]["status"], "completed_without_memory")


if __name__ == "__main__":
    unittest.main()
