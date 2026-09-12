import requests
from config import OLLAMA_CHAT_URL, OLLAMA_EMBED_URL, CHAT_MODEL, EMBED_MODEL


class OllamaClient:
    def __init__(self, chat_model=CHAT_MODEL, embed_model=EMBED_MODEL):
        self.chat_model = chat_model
        self.embed_model = embed_model

    def chat(self, messages, json_mode=False, temperature=0.2):
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": self.embed_model, "input": texts},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["embeddings"]
