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
from unittest.mock import MagicMock, patch

import agent_loop
import tools
from conversation import Convo
from memory import (
    MEMORY_PROMPT_PATH,
    RunRecorder,
    make_memory,
    normalize_review,
    render_memory_prompt,
    save_memory,
    verification_passed,
)
from memory_search import (
    MemoryVectorStore,
    guidance_message,
    memory_save_decision,
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
    def test_duplicate_exact_task_memory_is_not_saved_again(self) -> None:
        candidate = {
            "task": "open chrome",
            "playbook": {
                "learned_target_mappings": [
                    {
                        "requested": "Chrome",
                        "effective": "Chromium",
                    }
                ]
            },
        }
        existing = [
            {
                "match_type": "exact_task",
                "task": "open chrome",
                "playbook": {
                    "learned_target_mappings": [
                        {
                            "requested": "Chrome / Google Chrome",
                            "effective": "Chromium",
                        }
                    ]
                },
            }
        ]

        should_save, reason = memory_save_decision(candidate, existing)

        self.assertFalse(should_save)
        self.assertIn("already has", reason)

    def test_new_target_mapping_is_saved_for_exact_task(self) -> None:
        candidate = {
            "task": "open browser",
            "playbook": {
                "learned_target_mappings": [
                    {
                        "requested": "browser",
                        "effective": "Firefox",
                    }
                ]
            },
        }
        existing = [
            {
                "match_type": "exact_task",
                "task": "open browser",
                "playbook": {
                    "learned_target_mappings": [
                        {
                            "requested": "browser",
                            "effective": "Chromium",
                        }
                    ]
                },
            }
        ]

        should_save, reason = memory_save_decision(candidate, existing)

        self.assertTrue(should_save)
        self.assertIn("new target mapping", reason)

    def test_direct_type_then_enter_gets_automatic_settle_wait(self) -> None:
        actions = [
            {"tool": "type_text", "arguments": {"text": "Chromium"}},
            {"tool": "key_press", "arguments": {"key": "Enter"}},
        ]

        with (
            patch("tools.call_tool", return_value={}),
            patch("tools.time.sleep") as sleep_mock,
        ):
            results = tools.run_actions_safe(actions)

        sleep_mock.assert_called_once_with(0.8)
        self.assertEqual(
            results[1]["automatic_wait_before"]["seconds"],
            0.8,
        )

    def test_explicit_wait_prevents_duplicate_type_settle_wait(self) -> None:
        actions = [
            {"tool": "type_text", "arguments": {"text": "Chromium"}},
            {"tool": "wait", "arguments": {"seconds": 1}},
            {"tool": "key_press", "arguments": {"key": "Enter"}},
        ]

        with (
            patch("tools.call_tool", return_value={}),
            patch("tools.time.sleep") as sleep_mock,
        ):
            results = tools.run_actions_safe(actions)

        sleep_mock.assert_not_called()
        self.assertNotIn("automatic_wait_before", results[2])

    def test_type_text_uses_human_readable_default_interval(self) -> None:
        fake_pg = SimpleNamespace(write=MagicMock())

        with patch("tools._pyautogui", return_value=fake_pg):
            result = tools.type_text("Chromium")

        fake_pg.write.assert_called_once_with("Chromium", interval=0.02)
        self.assertEqual(result["interval"], 0.02)

    def test_mouse_click_glides_before_clicking(self) -> None:
        events = []

        class FakePyAutoGUI:
            easeInOutQuad = object()

            def position(self):
                return SimpleNamespace(x=0, y=0)

            def moveTo(self, x, y, **kwargs):
                events.append(("move", x, y, kwargs))

            def click(self, **kwargs):
                events.append(("click", kwargs))

        with patch("tools._pyautogui", return_value=FakePyAutoGUI()):
            result = tools.mouse_click("left", 900, 450)

        self.assertEqual(events[0][0], "move")
        self.assertEqual(events[1], ("click", {"button": "left"}))
        self.assertGreater(events[0][3]["duration"], 0)
        self.assertIn("tween", events[0][3])
        self.assertEqual(result["movement_duration"], events[0][3]["duration"])

    def test_longer_mouse_moves_take_longer(self) -> None:
        short = tools._movement_duration(0, 0, 100, 0)
        long = tools._movement_duration(0, 0, 1000, 0)

        self.assertGreater(long, short)
        self.assertGreaterEqual(short, 0.12)
        self.assertLessEqual(long, 0.7)

    def test_memory_prompt_is_loaded_from_jinja_template(self) -> None:
        rendered = render_memory_prompt()

        self.assertTrue(MEMORY_PROMPT_PATH.exists())
        self.assertIn("You verify and distill", rendered)
        self.assertIn('"learned_target_mappings"', rendered)

    def test_no_change_observation_forbids_repeated_blocked_clicks(self) -> None:
        text = agent_loop._observe_text(
            task="open chrome",
            step=3,
            previous_reply={
                "reason": "Click Chromium.",
                "actions": [
                    {
                        "tool": "mouse_double_click",
                        "arguments": {"button": "left", "x": 2000, "y": 500},
                    }
                ],
                "done": False,
            },
            tool_results=[],
            shot={"width": 3456, "height": 2234},
            mouse={"x": 2000, "y": 500},
            screen_change={"changed": False, "mean_delta": 0.02},
        )
        observation = json.loads(text)

        self.assertIn("Do not repeat", observation["last_action_outcome"])
        self.assertIn("modal", observation["last_action_outcome"])
        self.assertIn("not actionable through a modal", observation["instruction"])

    def test_vector_store_ranks_semantically_matching_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memories = root / "memories"
            memories.mkdir()
            chrome_memory = {
                "run_id": "chrome-run",
                "task": "open chrome",
                "verification": {"success": True, "confidence": 0.9},
                "playbook": {
                    "task_signature": "Open the Chrome-family browser",
                    "applicability": "Use when Chrome means installed Chromium.",
                    "learned_target_mappings": [
                        {
                            "requested": "Chrome",
                            "effective": "Chromium",
                            "relationship": "installed substitute",
                        }
                    ],
                    "preferred_plan": [],
                    "fallbacks": [],
                    "environment_facts": [],
                },
            }
            calculator_memory = {
                "run_id": "calculator-run",
                "task": "open calculator",
                "verification": {"success": True, "confidence": 0.9},
                "playbook": {
                    "task_signature": "Open Calculator",
                    "applicability": "Use for arithmetic tasks.",
                    "learned_target_mappings": [],
                    "preferred_plan": [],
                    "fallbacks": [],
                    "environment_facts": [],
                },
            }
            (memories / "chrome.json").write_text(json.dumps(chrome_memory))
            (memories / "calculator.json").write_text(json.dumps(calculator_memory))

            calls = []

            def embed(texts: list[str]) -> list[list[float]]:
                calls.append(list(texts))
                vectors = []
                for text in texts:
                    lowered = text.lower()
                    if "chrome" in lowered or "chromium" in lowered:
                        vectors.append([1.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0])
                return vectors

            store = MemoryVectorStore(
                memories_dir=memories,
                index_path=root / "index.sqlite3",
                embed_function=embed,
            )
            first = store.search("please open chrome", minimum_score=0.8)
            second = store.search("please open chrome", minimum_score=0.8)

            self.assertEqual([match["run_id"] for match in first], ["chrome-run"])
            self.assertEqual(second[0]["run_id"], "chrome-run")
            self.assertEqual(first[0]["match_type"], "exact_task")
            self.assertEqual(len(calls), 2)
            self.assertIn("retrieved_memory_guidance", guidance_message(first))

    def test_exact_task_memory_bypasses_vector_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memories = root / "memories"
            memories.mkdir()
            memory = {
                "run_id": "chrome-run",
                "task": "open chrome",
                "verification": {"success": True, "confidence": 0.96},
                "playbook": {
                    "task_signature": "Open a browser",
                    "applicability": "Browser requests",
                    "learned_target_mappings": [],
                    "preferred_plan": [],
                    "fallbacks": [],
                    "environment_facts": [],
                },
            }
            (memories / "chrome.json").write_text(json.dumps(memory))
            embedding_calls = []

            def embed(texts: list[str]) -> list[list[float]]:
                embedding_calls.append(list(texts))
                return [[1.0, 0.0] for _ in texts]

            store = MemoryVectorStore(
                memories_dir=memories,
                index_path=root / "index.sqlite3",
                embed_function=embed,
            )
            matches = store.search("open chrome", minimum_score=0.99)

            self.assertEqual(matches[0]["run_id"], "chrome-run")
            self.assertEqual(matches[0]["score"], 1.0)
            self.assertEqual(matches[0]["match_type"], "exact_task")
            self.assertEqual(len(embedding_calls), 1)

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

    def test_extra_memory_guidance_is_ephemeral(self) -> None:
        captured_messages = []
        fake_litellm = types.ModuleType("litellm")

        def completion(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                usage={},
            )

        fake_litellm.completion = completion
        fake_litellm.stream_chunk_builder = lambda chunks, messages: None
        convo = Convo(system="system")
        convo.user("task")

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            convo.complete(
                model="test-model",
                api_key="test-key",
                extra_messages=[{"role": "user", "content": "retrieved guidance"}],
            )

        self.assertEqual(captured_messages[-1]["content"], "retrieved guidance")
        self.assertNotIn("retrieved guidance", [message["content"] for message in convo.messages])

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
                patch(
                    "agent_loop.MemoryVectorStore",
                    return_value=SimpleNamespace(search=MagicMock(return_value=[])),
                ),
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
                    "agent_loop.MemoryVectorStore",
                    return_value=SimpleNamespace(search=MagicMock(return_value=[])),
                ),
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

    def test_agent_loop_reuses_exact_memory_instead_of_saving_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing_path = str(root / "memories" / "existing.json")
            exact_match = {
                "score": 1.0,
                "match_type": "exact_task",
                "path": existing_path,
                "run_id": "existing-run",
                "task": "open chrome",
                "verification": {"success": True, "confidence": 0.96},
                "playbook": {
                    "learned_target_mappings": [
                        {
                            "requested": "Chrome / Google Chrome",
                            "effective": "Chromium",
                        }
                    ],
                    "preferred_plan": [],
                },
            }

            def recorder_factory(task: str) -> RunRecorder:
                return RunRecorder(task, directory=root / "runs")

            with (
                patch("agent_loop.RunRecorder", side_effect=recorder_factory),
                patch(
                    "agent_loop.MemoryVectorStore",
                    return_value=SimpleNamespace(
                        search=MagicMock(return_value=[exact_match]),
                    ),
                ),
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
                        "reason": "Chromium is open.",
                        "actions": [],
                        "done": True,
                    },
                ),
                patch("agent_loop.review_completed_run", return_value=successful_review()),
                patch("agent_loop.save_memory") as save_mock,
            ):
                result = agent_loop.run_agent_loop(
                    "open chrome",
                    model="test-model",
                    api_key="test-key",
                    max_steps=1,
                    verbose=False,
                )

            self.assertTrue(result.memory_verified)
            self.assertEqual(result.memory_path, existing_path)
            save_mock.assert_not_called()
            run_events = [
                json.loads(line)
                for line in Path(result.run_path).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(run_events[-1]["status"], "memory_reused")

    def test_agent_loop_searches_memory_before_every_model_decision(self) -> None:
        search = MagicMock(
            return_value=[
                {
                    "score": 0.91,
                    "run_id": "prior-run",
                    "task": "open chrome",
                    "verification": {"success": True, "confidence": 0.9},
                    "playbook": {"preferred_plan": []},
                }
            ]
        )
        replies = iter(
            [
                {"reason": "Act.", "actions": [], "done": False},
                {"reason": "Done.", "actions": [], "done": True},
            ]
        )
        guidance_seen = []

        def complete_json(*args, **kwargs):
            guidance_seen.append(kwargs.get("memory_guidance"))
            return next(replies)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def recorder_factory(task: str) -> RunRecorder:
                return RunRecorder(task, directory=root / "runs")

            with (
                patch("agent_loop.RunRecorder", side_effect=recorder_factory),
                patch(
                    "agent_loop.MemoryVectorStore",
                    return_value=SimpleNamespace(search=search),
                ),
                patch(
                    "agent_loop.observe_screen",
                    return_value=(
                        {"path": None, "width": 100, "height": 100},
                        {"x": 1, "y": 2},
                        None,
                    ),
                ),
                patch("agent_loop._complete_json", side_effect=complete_json),
                patch("agent_loop.run_actions_safe", return_value=[]),
                patch(
                    "agent_loop.review_completed_run",
                    return_value={
                        "verification": {
                            "success": False,
                            "confidence": 0.2,
                            "summary": "Not verified.",
                        },
                        "playbook": {},
                    },
                ),
            ):
                agent_loop.run_agent_loop(
                    "open chrome",
                    model="test-model",
                    api_key="test-key",
                    max_steps=2,
                    verbose=False,
                )

        self.assertEqual(search.call_count, 2)
        self.assertTrue(all("retrieved_memory_guidance" in item for item in guidance_seen))


if __name__ == "__main__":
    unittest.main()
