MMFCUA - Muscle memmory for computer use agents

Small macOS computer-use agent experiment.

It takes a short task, looks at the screen, asks an LLM for the next primitive
action, runs it with pyautogui, then looks again.

This is for simple UI tasks only. Not a general desktop assistant.

Setup:

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=...
LITELLM_MODEL=gpt-4.1-nano
LITELLM_PROMPT_CACHE_RETENTION=24h
```

Run:

```bash
python main.py
```

Example tasks:

```text
open firefox and search weather in rome
open downloads folder
find keyboard settings
```
