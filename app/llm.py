from langchain_ollama import ChatOllama
from config import LOCAL_MODEL,OLLAMA_BASE_URL

local_llm = ChatOllama(
    model=LOCAL_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_predict=512,
    reasoning=False
)

print(f"Connected to Ollama model: {LOCAL_MODEL}")
print(f"Ollama URL: {OLLAMA_BASE_URL}")

test_response = local_llm.invoke(
    "Explain machine learning in one sentence."
)

print("CONTENT:")
print(test_response.content)

print("\nADDITIONAL KWARGS:")
print(test_response.additional_kwargs)