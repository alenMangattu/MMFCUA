import os

from dotenv import load_dotenv

from agent_loop import run_agent_loop
from prompt_ui import gettextfromui


def main() -> None:
    print("[main] loading environment", flush=True)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LITELLM_MODEL", "gpt-5.5")
    print(f"[main] model={model!r}", flush=True)

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or export it in your shell.")

    print("[main] opening task UI", flush=True)
    text = gettextfromui()
    print(f"[main] task received: {text!r}", flush=True)

    result = run_agent_loop(text, model=model, api_key=api_key, verbose=True)
    print(f"[main] final done={result.done} steps={result.steps}", flush=True)
    print(f"[main] final reply={result.final_reply}", flush=True)


if __name__ == "__main__":
    main()
