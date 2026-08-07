#!/usr/bin/env python3
"""G4 stage 2 -- embed every corpus document once and cache the vectors.

Runs on the HOST, not in the test image: ollama is published on 127.0.0.1:11434
only (the loopback-only-ports invariant), so a container in this compose project
cannot reach it. Only the standard library and numpy are needed here.

Document vector = MEAN OF CHUNK VECTORS, not a vector of the truncated page.
That is decision E5's regime and the one the TypeScript-era design will actually
use; evaluating the current whole-page-truncated regime would be measuring an
input that is scheduled for deletion. Chunk size/overlap/cap are taken verbatim
from the repo's own chunker (`services/ai-engine/app/chatbot/rag.py`:
CHUNK_SIZE=4000, CHUNK_OVERLAP=400, MAX_CHUNKS_PER_DOCUMENT=20).

Every chunk embedding is cached on disk keyed by sha256(model_id + chunk text),
so re-running any stage costs nothing and every arm sees byte-identical vectors.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# Verbatim from services/ai-engine/app/chatbot/rag.py.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400
MAX_CHUNKS_PER_DOCUMENT = 20


def chunk_ranges(content: str) -> list[tuple[int, int]]:
    """The repo's own chunker, copied so the eval cannot drift from it silently."""
    if not content:
        return [(0, 0)]
    step = CHUNK_SIZE - CHUNK_OVERLAP
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(content) and len(ranges) < MAX_CHUNKS_PER_DOCUMENT:
        end = min(start + CHUNK_SIZE, len(content))
        ranges.append((start, end))
        if end == len(content):
            break
        start += step
    return ranges


class VectorCache:
    """sha256(model, text) -> float32 vector, in a single sqlite file."""

    def __init__(self, path: Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, dim INT, v BLOB)"
        )
        self._db.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get(self, k: str):
        with self._lock:
            row = self._db.execute("SELECT dim, v FROM vec WHERE k=?", (k,)).fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32, count=dim)

    def put(self, k: str, vec: np.ndarray) -> None:
        b = np.asarray(vec, dtype=np.float32).tobytes()
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO vec VALUES (?,?,?)", (k, len(vec), b)
            )
            self._db.commit()


def embed_one(host: str, model: str, text: str, retries: int = 4) -> np.ndarray:
    payload = json.dumps({"model": model, "prompt": text}).encode()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{host}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            vec = np.asarray(data["embedding"], dtype=np.float32)
            if vec.size == 0 or not np.isfinite(vec).all():
                raise ValueError("embedding endpoint returned a degenerate vector")
            return vec
        except Exception as exc:  # noqa: BLE001 -- retried, then raised
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"embedding failed after {retries} attempts: {last}")


def model_fingerprint(host: str, model: str) -> dict:
    """Record exactly which weights produced these vectors."""
    req = urllib.request.Request(
        f"{host}/api/show",
        data=json.dumps({"name": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        info = json.load(resp)
    tags = json.load(urllib.request.urlopen(f"{host}/api/tags", timeout=30))
    digest = ""
    for m in tags.get("models", []):
        if m["name"] == model or m["name"].split(":")[0] == model.split(":")[0]:
            digest = m.get("digest", "")
            model = m["name"]
            break
    return {
        "provider": "ollama (local)",
        "model_id": model,
        "digest": digest,
        "details": info.get("details", {}),
        "parameters": info.get("parameters", ""),
        "cost": "0.00 (local, no metered provider used)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("G4_WORK", "/tmp/g4-clustering"))
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    work = Path(args.work)
    corpus_dir = work / "corpus"
    out_dir = work / "vectors"
    out_dir.mkdir(parents=True, exist_ok=True)

    fp = model_fingerprint(args.ollama, args.model)
    (work / "embedding_model.json").write_text(json.dumps(fp, indent=2))
    print(f"model: {fp['model_id']}  digest={fp['digest'][:16]}")

    cache = VectorCache(work / "vector_cache.sqlite")

    for path in sorted(corpus_dir.glob("*.json")):
        corpus = json.loads(path.read_text())
        docs = corpus["docs"]
        # Chunk every doc; collect the unique chunk texts that need embedding.
        doc_chunks: list[list[str]] = []
        for d in docs:
            text = (d["title"] + "\n\n" + d["text"]).strip()
            doc_chunks.append([text[s:e] for s, e in chunk_ranges(text)])

        wanted: dict[str, str] = {}
        for chunks in doc_chunks:
            for c in chunks:
                k = VectorCache.key(fp["model_id"], c)
                if cache.get(k) is None:
                    wanted[k] = c

        n_chunks = sum(len(c) for c in doc_chunks)
        multi = sum(1 for c in doc_chunks if len(c) > 1)
        print(
            f"{corpus['name']}: {len(docs)} docs, {n_chunks} chunks "
            f"({multi} docs >1 chunk), {len(wanted)} to embed",
            flush=True,
        )

        if wanted:
            items = list(wanted.items())
            done = 0
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
                futs = {
                    ex.submit(embed_one, args.ollama, args.model, text): k
                    for k, text in items
                }
                for fut in concurrent.futures.as_completed(futs):
                    cache.put(futs[fut], fut.result())
                    done += 1
                    if done % 200 == 0 or done == len(items):
                        rate = done / max(1e-6, time.time() - t0)
                        print(f"  {done}/{len(items)}  {rate:.1f}/s", flush=True)

        dim = None
        mats = []
        for chunks in doc_chunks:
            vecs = [cache.get(VectorCache.key(fp["model_id"], c)) for c in chunks]
            if any(v is None for v in vecs):
                raise RuntimeError("cache miss after embedding pass")
            v = np.mean(np.vstack(vecs), axis=0)
            dim = v.shape[0]
            mats.append(v)
        X = np.vstack(mats).astype(np.float32)
        assert X.shape[0] == len(docs)

        np.savez_compressed(
            out_dir / f"{corpus['name']}.npz",
            X=X,
            ids=np.array([d["id"] for d in docs]),
            true_group=np.array([d["true_group"] for d in docs]),
            kind=np.array([d["kind"] for d in docs]),
            inc=np.array([d["inc"] for d in docs], dtype=np.int32),
            url=np.array([d["url"] for d in docs]),
            title=np.array([d["title"] for d in docs]),
        )
        stats = {
            "n_docs": len(docs),
            "n_chunks": n_chunks,
            "docs_multi_chunk": multi,
            "dim": int(dim),
        }
        (out_dir / f"{corpus['name']}.stats.json").write_text(json.dumps(stats))
        print(f"  wrote {out_dir / (corpus['name'] + '.npz')}  dim={dim}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
