# MMFCUA - Muscle memory for computer-use agents

Small macOS computer-use agent experiment.

It takes a short task, looks at the screen, asks an LLM for the next primitive
action, runs it with pyautogui, then looks again.

This is for simple UI tasks only. Not a general desktop assistant.

> [!WARNING]
> MMFCUA currently supports macOS only. The prompt window uses Apple's AppKit,
> screenshots use macOS system tools, and the agent expects macOS keyboard
> shortcuts. Windows and Linux users will need to replace these platform-specific
> parts before running the project.

## Setup

The native prompt UI is compiled with `swiftc` the first time the project runs.
Install the Xcode Command Line Tools if they are not already available:

```bash
xcode-select --install
```

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=...
LITELLM_MODEL=gpt-4.1-nano
LITELLM_PROMPT_CACHE_RETENTION=24h
```

## macOS permissions

MMFCUA needs permission to see and control the desktop. In **System Settings >
Privacy & Security**, enable:

- **Accessibility** for the terminal or IDE that runs `python main.py`.
- **Screen Recording** for the same terminal or IDE.

macOS may display a permission prompt on the first run. After changing either
permission, fully quit and reopen the terminal or IDE before trying again.
Without these permissions, screenshots may be blank and mouse or keyboard
actions may silently fail.

## Run

```bash
python main.py
```

The project builds and opens a small native Swift/AppKit task window. If it does
not open:

- Confirm `swiftc --version` works. If it does not, install the Xcode Command
  Line Tools with `xcode-select --install`.
- If macOS blocks the generated `.build/prompt_input` binary, allow it in
  **System Settings > Privacy & Security**, then run the command again.
- Run the project from a normal logged-in macOS desktop session. The prompt UI
  and computer-control features will not work correctly over a headless shell.

## Example tasks

```text
open firefox and search weather in rome
open downloads folder
find keyboard settings
```
