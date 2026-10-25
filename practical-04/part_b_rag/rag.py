"""Retrieval-Augmented Generation building blocks: chunking, embeddings, vector stores, LLM calls."""
from __future__ import annotations

import re
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ABBREVIATIONS = {"dr", "prof", "mr", "ms", "mrs", "st", "vs"}


# --------------------------------------------------------------------------- documents
@dataclass
class Doc:
    doc_id: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str

    @property
    def embed_text(self) -> str:
        """Text that is embedded: the document title is prepended as context for short chunks."""
        return f"{self.title}. {self.text}"

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def load_documents(kb_dir: str | Path) -> list[Doc]:
    docs = []
    for path in sorted(Path(kb_dir).glob("*.txt")):
        title, _, body = path.read_text(encoding="utf-8").partition("\n")
        docs.append(Doc(path.stem, title.strip(), body.strip()))
    return docs


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that does not break after titles such as 'Dr.' or 'Prof.'."""
    pieces = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"])', text.strip())
    sentences: list[str] = []
    for piece in pieces:
        if sentences and sentences[-1].rstrip(".").split()[-1].lower() in ABBREVIATIONS:
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return sentences


def chunk_document(doc: Doc, max_words: int) -> list[Chunk]:
    """Greedily pack whole sentences into chunks of at most ``max_words`` words."""
    chunks: list[list[str]] = []
    current: list[str] = []
    count = 0
    for sentence in split_sentences(doc.text):
        n = len(sentence.split())
        if current and count + n > max_words:
            chunks.append(current)
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        chunks.append(current)
    return [Chunk(f"{doc.doc_id}#{i}", doc.doc_id, doc.title, " ".join(c)) for i, c in enumerate(chunks)]


def build_chunks(docs: list[Doc], max_words: int) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, max_words)]


# --------------------------------------------------------------------------- embeddings
class Embedder:
    """Sentence-Transformers wrapper that returns L2-normalised vectors."""

    def __init__(self, key: str, model_id: str, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.key, self.model_id, self.query_prefix = key, model_id, query_prefix
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_id, device="cpu")
        self.load_s = time.perf_counter() - t0
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = dim_fn()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True, convert_to_numpy=True)[0]


# --------------------------------------------------------------------------- vector stores
class ChromaStore:
    """ChromaDB persistent collection using an HNSW index with cosine distance."""

    name = "chroma"

    def __init__(self, path: str | Path, collection: str):
        import chromadb
        from chromadb.config import Settings

        shutil.rmtree(path, ignore_errors=True)
        self.path = Path(path)
        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        try:
            self.col = self.client.create_collection(
                collection, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
        except TypeError:  # older chromadb releases
            self.col = self.client.create_collection(
                collection, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "title": c.title} for c in chunks],
        )

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        res = self.col.query(query_embeddings=[vector.tolist()], n_results=k)
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def disk_mb(self) -> float:
        return sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file()) / 1e6


class FaissStore:
    """FAISS exact inner-product index (equals cosine similarity for normalised vectors)."""

    name = "faiss"

    def __init__(self, dim: int):
        import faiss

        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.ids.extend(c.chunk_id for c in chunks)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        scores, idx = self.index.search(np.ascontiguousarray(vector[None, :], dtype="float32"), k)
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


class BM25Store:
    """Lexical baseline (Okapi BM25) over the same chunks."""

    name = "bm25"

    def __init__(self):
        self.ids: list[str] = []
        self.bm25 = None

    @staticmethod
    def tokenise(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def add(self, chunks: list[Chunk], _vectors=None) -> None:
        from rank_bm25 import BM25Okapi

        self.ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([self.tokenise(c.embed_text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(self.tokenise(query))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


# --------------------------------------------------------------------------- retrieval metrics
def doc_of(chunk_id: str) -> str:
    return chunk_id.split("#")[0]


def first_relevant_rank(ranked_chunk_ids: list[str], gold_docs: list[str]) -> int | None:
    """1-based rank of the first chunk that belongs to a gold document, else None."""
    for rank, cid in enumerate(ranked_chunk_ids, start=1):
        if doc_of(cid) in gold_docs:
            return rank
    return None


def summarise_ranks(ranks: list[int | None], ks=(1, 3, 5, 10)) -> dict:
    n = len(ranks)
    out = {f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 4) for k in ks}
    out["mrr@10"] = round(sum(1 / r for r in ranks if r is not None and r <= 10) / n, 4)
    out["n_queries"] = n
    return out


def median_ms(fn, repeats: int = 30) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 3)


# --------------------------------------------------------------------------- generation
SYSTEM_RAG = (
    "You are a helpful assistant for the Zephyr Institute of Technology (ZIT). "
    "Answer the question using ONLY the context passages provided. "
    "If the passages do not contain the answer, reply exactly: I don't know based on the provided documents. "
    "Be concise (one or two sentences) and end with the source id(s) in square brackets, for example [library]."
)
SYSTEM_CLOSED = "You are a helpful assistant. Answer the question concisely in one or two sentences."

ABSTAIN_PATTERN = re.compile(
    r"(?:don'?t|do not|doesn'?t|does not) (?:know|have|mention|specify|provide|contain|say)"
    r"|(?:isn'?t|is not|aren'?t|are not|wasn'?t|not) (?:mentioned|provided|specified|stated|available|included"
    r"|listed|found|familiar|aware|sure|publicly)"
    r"|no (?:information|mention|details?|data|record|specific)"
    r"|(?:cannot|can'?t|couldn'?t|could not|unable to) (?:find|answer|determine|provide|verify|confirm|locate|access)"
    r"|insufficient|unclear",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    return ABSTAIN_PATTERN.search(answer.replace("’", "'")) is not None


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.doc_id}] {c.title}: {c.text}" for c in chunks)


def ask_ollama(client, model: str, system: str, user: str, options: dict) -> dict:
    """One chat turn; returns the answer text plus Ollama's own timing counters."""
    t0 = time.perf_counter()
    resp = client.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options=options,
    )
    wall = time.perf_counter() - t0
    get = lambda name: getattr(resp, name, None) or 0  # noqa: E731
    eval_s = get("eval_duration") / 1e9
    return {
        "answer": resp.message.content.strip(),
        "wall_s": round(wall, 3),
        "prompt_tokens": get("prompt_eval_count"),
        "gen_tokens": get("eval_count"),
        "prompt_eval_s": round(get("prompt_eval_duration") / 1e9, 3),
        "gen_s": round(eval_s, 3),
        "tok_per_s": round(get("eval_count") / eval_s, 3) if eval_s > 0 else 0.0,
        "load_s": round(get("load_duration") / 1e9, 3),
    }
