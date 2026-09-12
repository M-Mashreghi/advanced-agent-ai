from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
EXA_SEARCH_URL = "https://api.exa.ai/search"

CHUNK_WORDS = 120
CHUNK_OVERLAP_WORDS = 25
RAG_CANDIDATES = 10
RAG_TOP_K = 3
DENSE_WEIGHT = 0.70
BM25_WEIGHT = 0.30
MAX_HISTORY_MESSAGES = 12
RECENT_MESSAGES_AFTER_SUMMARY = 6
MAX_GAME_TURNS = 20
