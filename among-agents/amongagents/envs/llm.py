import os
import requests

MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")


def generate_local(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 160,
            }
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def generate_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()

    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"].strip()

    raise RuntimeError(f"OpenRouter response missing choices: {data}")


def generate(prompt: str) -> str:
    try:
        return generate_local(prompt)
    except Exception as local_error:
        if OPENROUTER_API_KEY:
            print(f"Local LLM generation failed, falling back to OpenRouter: {local_error}")
            return generate_openrouter(prompt)
        raise
