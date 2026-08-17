import httpx
from langchain_ollama import ChatOllama
from config import LOCAL_MODEL, OLLAMA_BASE_URL


def check_ollama_connection(base_url: str = OLLAMA_BASE_URL, timeout: float = 2.0) -> bool:
    """Cheap reachability check against the Ollama server (no model call)."""
    try:
        httpx.get(base_url, timeout=timeout)
        return True
    except httpx.ConnectError:
        return False


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

# Only fire a real (slower) test generation when llm.py is run directly,
# not on every import from rag.py / other modules.
if __name__ == "__main__":
    test_response = local_llm.invoke(
        "Explain machine learning in one sentence."
    )

    print("CONTENT:")
    print(test_response.content)

    print("\nADDITIONAL KWARGS:")
    print(test_response.additional_kwargs)