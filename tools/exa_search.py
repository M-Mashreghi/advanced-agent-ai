import requests
import numpy as np
from config import EXA_API_KEY, EXA_SEARCH_URL


class ExaSearch:
    def __init__(self, llm, api_key=EXA_API_KEY):
        self.llm = llm
        self.api_key = api_key

    def available(self):
        return bool(self.api_key)

    def search(self, query, num_results=10, top_k=3):
        if not self.api_key:
            raise RuntimeError("EXA_API_KEY is missing. Put it in your .env file.")

        r = requests.post(
            EXA_SEARCH_URL,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "type": "auto",
                "numResults": num_results,
                "contents": {"highlights": True},
            },
            timeout=60,
        )
        r.raise_for_status()
        raw_results = r.json().get("results", [])

        prepared = []
        for item in raw_results:
            highlights = item.get("highlights") or []
            if isinstance(highlights, str):
                highlights = [highlights]
            text = "\n".join(highlights).strip() or str(item.get("text", ""))[:1800]
            prepared.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "published_date": item.get("publishedDate", ""),
                "text": text,
            })

        if not prepared:
            return []

        q = np.asarray(self.llm.embed(query)[0], dtype="float32")
        vecs = np.asarray(self.llm.embed([f'{x["title"]}\n{x["text"]}' for x in prepared]), dtype="float32")
        qn = np.linalg.norm(q) or 1.0
        for item, vec in zip(prepared, vecs):
            denom = qn * (np.linalg.norm(vec) or 1.0)
            item["score"] = float(np.dot(q, vec) / denom)

        prepared.sort(key=lambda x: x["score"], reverse=True)
        selected = prepared[:top_k]

        for item in selected:
            prompt = f"""
Summarize this web result for the query in at most 3 concise sentences.
Preserve factual details and do not invent information.

Query: {query}
Title: {item["title"]}
Content: {item["text"]}
"""
            try:
                item["summary"] = self.llm.chat([{"role":"user","content":prompt}], temperature=0.1).strip()
            except Exception:
                item["summary"] = item["text"]

        return selected

    @staticmethod
    def build_context(results):
        return "\n\n".join(
            f'[Web {i}: {item["title"]}]\nURL: {item["url"]}\nDate: {item["published_date"]}\n{item["summary"]}'
            for i, item in enumerate(results, start=1)
        )
