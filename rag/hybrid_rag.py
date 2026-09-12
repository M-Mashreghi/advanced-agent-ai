import hashlib
import json
import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from config import CACHE_DIR, CHUNK_WORDS, CHUNK_OVERLAP_WORDS, RAG_CANDIDATES, RAG_TOP_K, DENSE_WEIGHT, BM25_WEIGHT


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def chunk_pages(pages):
    chunks, chunk_id = [], 0
    for page in pages:
        words = page["text"].split()
        step = max(1, CHUNK_WORDS - CHUNK_OVERLAP_WORDS)
        for start in range(0, len(words), step):
            part = words[start:start + CHUNK_WORDS]
            if not part:
                continue
            chunks.append({
                "id": chunk_id,
                "filename": page["filename"],
                "page": page["page"],
                "text": " ".join(part),
            })
            chunk_id += 1
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


def corpus_hash(uploaded_files):
    digest = hashlib.sha256()
    for file in sorted(uploaded_files, key=lambda x: x.name):
        digest.update(file.name.encode("utf-8"))
        digest.update(file.getvalue())
    return digest.hexdigest()[:20]


def _minmax(values):
    arr = np.asarray(values, dtype="float32")
    if len(arr) == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.ones_like(arr) if hi != 0 else np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class HybridRAG:
    def __init__(self, llm):
        self.llm = llm
        self.chunks = []
        self.index = None
        self.bm25 = None
        self.corpus_id = None

    def ready(self):
        return self.index is not None and bool(self.chunks)

    def _paths(self, corpus_id):
        folder = CACHE_DIR / corpus_id
        return folder, folder / "faiss.index", folder / "chunks.json"

    def load(self, corpus_id):
        folder, index_path, chunks_path = self._paths(corpus_id)
        if not index_path.exists() or not chunks_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.corpus_id = corpus_id
        return True

    def build(self, chunks, corpus_id):
        if not chunks:
            raise ValueError("No readable chunks were created from the PDFs.")
        texts = [c["text"] for c in chunks]
        vectors = []
        for i in range(0, len(texts), 32):
            vectors.extend(self.llm.embed(texts[i:i + 32]))
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)

        # Assignment-compatible persistent IndexFlatL2.
        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)

        self.chunks = chunks
        self.index = index
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.corpus_id = corpus_id

        folder, index_path, chunks_path = self._paths(corpus_id)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query, candidate_k=RAG_CANDIDATES, top_k=RAG_TOP_K):
        if not self.ready():
            return []

        q = np.asarray([self.llm.embed(query)[0]], dtype="float32")
        faiss.normalize_L2(q)
        k = min(candidate_k, len(self.chunks))
        distances, indices = self.index.search(q, k)

        dense = {}
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                dense[int(idx)] = 1.0 / (1.0 + float(distance))

        bm25_raw = self.bm25.get_scores(tokenize(query))
        bm25_norm = _minmax(bm25_raw)
        candidate_ids = set(dense)
        candidate_ids.update(int(i) for i in np.argsort(bm25_raw)[::-1][:k])

        results = []
        for idx in candidate_ids:
            d = dense.get(idx, 0.0)
            s = float(bm25_norm[idx])
            h = DENSE_WEIGHT * d + BM25_WEIGHT * s
            item = dict(self.chunks[idx])
            item.update({"dense_score": d, "bm25_score": float(bm25_raw[idx]), "hybrid_score": float(h)})
            results.append(item)

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return self._rerank(query, results[:candidate_k], top_k)

    def _rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        compact = [{"id":i, "source":f'{c["filename"]} page {c["page"]}', "text":c["text"][:1200]} for i,c in enumerate(candidates)]
        prompt = f"""
Rank passages by usefulness for answering the query.
Return ONLY JSON: {{"ranking":[0,2,1]}}

Query: {query}
Passages: {json.dumps(compact, ensure_ascii=False)}
"""
        try:
            raw = self.llm.chat([{"role":"user","content":prompt}], json_mode=True, temperature=0.0)
            ranking = json.loads(raw).get("ranking", [])
            chosen, used = [], set()
            for idx in ranking:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in used:
                    chosen.append(candidates[idx]); used.add(idx)
                    if len(chosen) == top_k:
                        break
            for idx, item in enumerate(candidates):
                if len(chosen) == top_k:
                    break
                if idx not in used:
                    chosen.append(item)
            return chosen[:top_k]
        except Exception:
            return candidates[:top_k]

    @staticmethod
    def build_context(results):
        return "\n\n".join(
            f'[Doc {i}: {item["filename"]}, page {item["page"]}]\n{item["text"]}'
            for i, item in enumerate(results, start=1)
        )
