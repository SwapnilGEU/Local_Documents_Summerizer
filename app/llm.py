import difflib

import httpx
from config import LOCAL_MODEL, OLLAMA_BASE_URL
from langchain_ollama import ChatOllama


def check_ollama_connection(
    base_url: str = OLLAMA_BASE_URL, timeout: float = 2.0
) -> bool:
    """Cheap reachability check against the Ollama server (no model call)."""
    try:
        httpx.get(base_url, timeout=timeout)
        return True
    except httpx.ConnectError:
        return False


def check_model_available(
    model: str = LOCAL_MODEL, base_url: str = OLLAMA_BASE_URL, timeout: float = 5.0
):
    """Is `model` pulled in Ollama? Returns (ok, installed_model_names)."""
    tags = httpx.get(f"{base_url}/api/tags", timeout=timeout).json().get("models", [])
    installed = sorted(m["name"] for m in tags)
    ok = model in installed or f"{model}:latest" in installed
    return ok, installed


local_llm = ChatOllama(
    model=LOCAL_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_predict=512,
    reasoning=False,
)

print(f"Configured Ollama model: {LOCAL_MODEL}")
print(f"Ollama URL: {OLLAMA_BASE_URL}")

if not check_ollama_connection():
    raise RuntimeError(
        f"Ollama server not reachable at {OLLAMA_BASE_URL}.\n"
        "Start it with `ollama serve`, or open the Ollama desktop app, "
        "then re-run this script."
    )

print("Ollama server is reachable.")

# Fail at startup (not halfway through a request) if the model name is wrong
# or the model was never pulled.
_model_ok, _installed = check_model_available()
if not _model_ok:
    _close = difflib.get_close_matches(LOCAL_MODEL, _installed, n=1)
    _hint = f" Did you mean '{_close[0]}'?" if _close else ""
    raise RuntimeError(
        f"Model '{LOCAL_MODEL}' is not installed in Ollama.{_hint}\n"
        f"Installed: {', '.join(_installed) or 'none'}\n"
        "Fix LOCAL_MODEL in app/config.py (or the LOCAL_MODEL env var), "
        f"or run `ollama pull {LOCAL_MODEL}`."
    )

print(f"Model '{LOCAL_MODEL}' is available.")

# Only fire a real (slower) test generation when llm.py is run directly,
# not on every import from rag.py / other modules.
if __name__ == "__main__":
    test_response = local_llm.invoke("Explain machine learning in one sentence.")

    print("CONTENT:")
    print(test_response.content)

    print("\nADDITIONAL KWARGS:")
    print(test_response.additional_kwargs)
