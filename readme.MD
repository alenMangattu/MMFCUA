# MMFCUA

**A computer-use agent that turns successful UI interactions into reusable
procedural memory.**

MMFCUA completes small macOS tasks through screenshots, mouse input, and
keyboard input. Its central concern is not simply getting an LLM to click the
right place once. It is deciding what should be learned from a successful run,
storing that knowledge in a durable form, and retrieving it when a similar task
appears later.

The result is a small research-oriented system for **verified agent memory**:

- every interaction is recorded as a factual execution trace;
- claimed success is reviewed against the final screen;
- only sufficiently credible runs become reusable memories;
- memories are stored as semantic playbooks rather than raw action replays;
- relevant playbooks are retrieved through lexical and vector search;
- the current screen always overrides historical experience.

MMFCUA is intentionally limited to short, visually verifiable desktop tasks. It
is not a general desktop assistant or a chat interface around an automation
script.

## The memory system

The main artifact produced by MMFCUA is not a conversation transcript. It is a
verified description of how a class of task was completed.

### 1. Record the run

Each task creates an append-only JSONL trace in `.runs/`. The trace contains the
initial observation, model decisions, executed actions, tool outcomes, screen
changes, and the final completion claim.

Recording is incremental, so evidence from a partial run survives a crash or
interruption.

### 2. Verify completion

An agent saying `done` is not treated as proof of success.

Once completion is claimed, a separate review pass receives the final
screenshot and a compact version of the trajectory. It evaluates whether the
requested outcome is visibly supported and assigns a confidence score. Runs
below `MMFCUA_MEMORY_MIN_CONFIDENCE` are retained as traces but are not promoted
to memory.

This separates **task execution** from **memory admission**. A weak or mistaken
completion claim does not automatically become future guidance.

### 3. Distill a playbook

A verified run is reduced to a small procedural representation containing:

- a task signature and applicability boundary;
- learned mappings between requested and effective UI targets;
- a preferred plan with preconditions and success checks;
- bounded fallback strategies;
- actions or assumptions to avoid;
- timing and environment facts that may require revalidation.

The playbook preserves reusable structure while discarding most run-specific
noise. Pointer coordinates are explicitly treated as historical evidence, never
as instructions to replay.

### 4. Store and index it

Verified playbooks are written as JSON documents in `.memories/`. Their
embeddings and source documents are maintained in a local SQLite index at
`.memory_index.sqlite3`.

The index is updated incrementally. Existing exact-task memories are reused
instead of duplicated unless a new run contributes a previously unseen target
mapping.

### 5. Retrieve relevant experience

Before each model decision, MMFCUA searches memory using:

- exact normalized task matching;
- lexical matching over task signatures and learned aliases;
- cosine similarity over embeddings.

Only results above the configured relevance threshold are injected into the
decision context. Retrieved memories are advisory: aliases, application state,
preconditions, and success criteria must be checked against the current
screenshot.

```text
                       successful run
                             |
                             v
task -> act -> observe -> completion claim
                             |
                             v
                  independent verification
                             |
                    confidence threshold
                       /           \
                  rejected       accepted
                     |              |
                  trace only     distill playbook
                                    |
                              store + embed
                                    |
                         retrieve on future tasks
```

## The execution loop

Memory supports a deliberately small computer-control loop:

1. Capture the screen and pointer position.
2. Retrieve relevant verified playbooks.
3. Ask a vision-capable model for a short batch of primitive actions.
4. Execute the actions with `pyautogui`.
5. Capture the screen again and measure visible change.
6. Continue, revise the strategy, or claim completion.

The model emits structured JSON rather than free-form instructions. Available
operations include pointer movement, clicks, dragging, scrolling, text entry,
keyboard shortcuts, and waits.

A lightweight image diff identifies action batches with no visible effect. The
next decision is then instructed to inspect blockers or change strategy instead
of repeating nearby clicks. The textual trajectory is retained in context, but
only the newest screenshot is sent to the model.

The LLM is therefore one component in the system: it proposes actions and
reviews outcomes. Trace persistence, admission policy, playbook structure,
retrieval, and execution safety are implemented locally.

## Project structure

```text
agent_loop.py       observation, retrieval, action, and verification lifecycle
memory.py           run recording, completion review, and playbook persistence
memory_search.py    SQLite vector index, retrieval, and memory deduplication
tools.py            constrained mouse and keyboard primitives
conversation.py     multimodal LiteLLM conversation state
prompt.j2           computer-use policy and action contract
memory_prompt.j2    verification and playbook extraction contract
prompt_ui.py        native task-input launcher
components/         Swift/AppKit prompt interface
```

## Requirements

- macOS
- Python and `pip`
- Xcode Command Line Tools
- an OpenAI API key

The task prompt is a small Swift/AppKit application compiled with `swiftc` on
its first run:

```bash
xcode-select --install
```

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=...
LITELLM_MODEL=gpt-5.4-mini
```

Start MMFCUA:

```bash
python main.py
```

A native prompt window appears near the bottom of the active display. Enter one
short task and press **Enter**. Press **Escape** to close it without submitting.

Example tasks:

```text
open Firefox and search for weather in Rome
open the Downloads folder
find the keyboard settings
```

Tasks work best when they have one concrete outcome that can be confirmed from
the screen.

## macOS permissions

In **System Settings > Privacy & Security**, grant the terminal or IDE running
`python main.py`:

- **Accessibility**
- **Screen Recording**

Fully quit and reopen the terminal or IDE after changing either permission.
Without them, screenshots may be blank and input actions may silently fail.

The current implementation assumes macOS screenshots, AppKit, and macOS
keyboard conventions. Windows and Linux support would require replacing those
platform-specific components.

## Configuration

The defaults are suitable for short interactive tasks. Override only the
settings relevant to an experiment:

```env
# Main and review models
LITELLM_MODEL=gpt-5.4-mini
MMFCUA_REVIEW_MODEL=gpt-5.4-mini
MMFCUA_REASONING_EFFORT=none
MMFCUA_VERBOSITY=low
MMFCUA_MAX_COMPLETION_TOKENS=800
MMFCUA_STREAM_MODEL_OUTPUT=1

# Memory admission and retrieval
MMFCUA_MEMORY_MIN_CONFIDENCE=0.75
MMFCUA_EMBEDDING_MODEL=text-embedding-3-small
MMFCUA_MEMORY_SEARCH_MIN_SCORE=0.72
MMFCUA_MEMORY_SEARCH_LIMIT=2

# Prompt caching
LITELLM_PROMPT_CACHE_RETENTION=24h

# Observation and input timing
MMFCUA_OBSERVATION_SETTLE_SECONDS=0.1
MMFCUA_MOUSE_PIXELS_PER_SECOND=1800
MMFCUA_MOUSE_MIN_DURATION=0.12
MMFCUA_MOUSE_MAX_DURATION=0.7
MMFCUA_TYPE_INTERVAL=0.02
MMFCUA_POST_TYPE_SETTLE_SECONDS=0.8
```

`OPENAI_SERVICE_TIER` is passed through when present.

## Operational notes

If the prompt window does not open:

- confirm that `swiftc --version` succeeds;
- install the Xcode Command Line Tools with `xcode-select --install`;
- allow `.build/prompt_input` in **System Settings > Privacy & Security** if
  macOS blocks it;
- run the project from a normal logged-in desktop session rather than a
  headless shell.

MMFCUA can move the pointer and type into the active interface. Use it in a
supervised desktop session and begin with low-risk tasks.
