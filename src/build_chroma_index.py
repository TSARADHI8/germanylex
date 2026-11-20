"""Build the persistent ChromaDB index for GermanyLex: 718 chunks + BGE-M3 embeddings + metadata.

What it stores (one Chroma record per chunk)
  id         chunk_id, e.g. "aufenthg_2004__16b__c02"
  embedding  BGE-M3 dense vector (1024 dims, unit length), computed exactly as in the evaluation
             (src/compare_chunk_embeddings.py): input "heading\\ntext", no prefix, max_length 1024
  document   the chunk text, unchanged
  metadata   section_id, law, law_title, area, section, heading, source_url, retrieval_date,
             chunk_index, n_chunks, paragraphs, split_method, char_start, char_end, token_count

The collection uses cosine distance (distance = 1 - cosine similarity) and NO Chroma embedding
function: vectors are always supplied by us, so Chroma never downloads or runs another model.

Reused code: HFEmbedder and MODELS (embedding_smoke_test.py); encode, load_chunks, passage_text,
sha256_of (compare_chunk_embeddings.py). Nothing in the evaluation scripts, chunk files, eval
file or legal records is modified (chunk files are hashed before and after).

Usage (repo root):
    python src/build_chroma_index.py                 # encode 718 chunks (~13 min on CPU), store, verify
    python src/build_chroma_index.py --verify-only   # re-open chroma_db/ in a new process and re-check
    python src/build_chroma_index.py --rebuild       # replace an existing collection
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_chunk_embeddings import encode, load_chunks, passage_text, sha256_of  # noqa: E402
from embedding_smoke_test import MODELS, HFEmbedder  # noqa: E402

ROOT = HERE.parent
CHUNK_DIR = ROOT / "data" / "processed_chunks"
DB_DIR = ROOT / "chroma_db"                       # already in .gitignore
REPORT = ROOT / "reports" / "chroma_index_build.json"
COLLECTION = "germanylex_chunks"

# Fixed to what was evaluated (dev and test). Deliberately not command-line options.
MODEL_KEY, MAX_LENGTH, BATCH_SIZE, PASSAGE_FORMAT = "bge_m3", 1024, 4, "heading_text"
META_FIELDS = ("section_id", "law", "law_title", "area", "section", "heading", "source_url", "retrieval_date",
               "chunk_index", "n_chunks", "split_method", "char_start", "char_end", "token_count")
ADD_BATCH = 200
# HNSW is an approximate index. With only 718 vectors a denser graph than Chroma's default costs nothing
# and removes the occasional miss of short, isolated chunks (tested: exact-search overlap 1.0000).
HNSW = {"space": "cosine", "max_neighbors": 48, "ef_construction": 400, "ef_search": 400}


# ------------------------------------------------------------------ helpers
def chunk_metadata(c: dict) -> dict:
    """Chroma metadata may only hold str/int/float/bool, so the paragraph list becomes a string."""
    meta = {}
    for k in META_FIELDS:
        v = c.get(k)
        if not isinstance(v, (str, int)) or isinstance(v, bool):
            sys.exit(f"{c.get('chunk_id')}: metadata field {k!r} must be a string or int, got {v!r}")
        meta[k] = v
    meta["paragraphs"] = ", ".join(c.get("paragraphs") or [])
    return meta


def file_hashes(chunk_dir: Path) -> dict:
    return {p.name: sha256_of(p) for p in sorted(chunk_dir.glob("*.json*"))}


def get_client(db_dir: Path):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(db_dir), settings=Settings(anonymized_telemetry=False))


def collection_names(client) -> list[str]:
    return [c if isinstance(c, str) else c.name for c in client.list_collections()]


def dir_size_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6, 1)


# ------------------------------------------------------------------ verification
def verify(db_dir: Path, name: str, chunks: list[dict]):
    """Re-open the database with a NEW client and compare everything with the chunk files."""
    checks = []

    def add(label, ok, detail=""):
        checks.append({"check": label, "ok": bool(ok), "detail": detail})

    client = get_client(db_dir)
    if name not in collection_names(client):
        add("collection exists", False, f"{name!r} not found in {db_dir}")
        return checks, {}
    col = client.get_collection(name, embedding_function=None)
    n = col.count()
    add("collection holds one record per chunk", n == len(chunks), f"{n} records, {len(chunks)} chunks")

    got = col.get(include=["documents", "metadatas", "embeddings"])
    ids = got["ids"]
    by_id = {i: k for k, i in enumerate(ids)}
    want_ids = [c["chunk_id"] for c in chunks]
    add("ids are exactly the chunk ids", len(ids) == len(set(ids)) and set(ids) == set(want_ids))

    docs_ok = meta_ok = 0
    bad = []
    for c in chunks:
        k = by_id.get(c["chunk_id"])
        if k is None:
            continue
        d_ok = got["documents"][k] == c["text"]
        m_ok = dict(got["metadatas"][k]) == chunk_metadata(c)
        docs_ok += d_ok
        meta_ok += m_ok
        if not (d_ok and m_ok):
            bad.append(c["chunk_id"])
    add("every document equals the chunk text", docs_ok == len(chunks), f"{docs_ok}/{len(chunks)}")
    add("every metadata record equals the chunk file", meta_ok == len(chunks),
        f"{meta_ok}/{len(chunks)}" + (f"; first mismatches: {bad[:3]}" if bad else ""))

    emb = np.asarray(got["embeddings"], dtype=np.float32)
    dim = MODELS[MODEL_KEY]["dim"]
    norms = np.linalg.norm(emb, axis=1) if emb.ndim == 2 else np.array([0.0])
    add(f"embeddings are {len(chunks)} x {dim}", emb.shape == (len(chunks), dim), f"shape {tuple(emb.shape)}")
    add("embeddings finite and unit length", bool(np.isfinite(emb).all()) and bool(np.allclose(norms, 1.0, atol=1e-3)),
        f"norm min {norms.min():.4f} max {norms.max():.4f}")
    add("no two chunks share the same vector", len({v.tobytes() for v in emb}) == len(emb))

    # Chroma's approximate search must agree with exact brute-force cosine search on the stored vectors
    res = col.query(query_embeddings=emb.tolist(), n_results=10, include=[])
    exact = np.argsort(-(emb @ emb.T), axis=1, kind="stable")[:, :10]
    recall = float(np.mean([len({ids[j] for j in exact[k]} & set(res["ids"][k])) / 10 for k in range(len(ids))]))
    top1 = sum(r[0] == i for r, i in zip(res["ids"], ids))
    in10 = sum(i in r for r, i in zip(res["ids"], ids))
    add("approximate search matches exact search (recall@10 >= 0.99)", recall >= 0.99,
        f"recall@10 {recall:.4f}; a vector finds its own chunk first for {top1}/{len(ids)}, in top-10 for {in10}/{len(ids)}")

    # metadata filters must work, they will be needed for law / area filtering later
    per_law_file = Counter(c["law"] for c in chunks)
    per_law_db = {law: len(col.get(where={"law": law}, include=[])["ids"]) for law in per_law_file}
    add("metadata filter works (per-law counts)", per_law_db == dict(per_law_file), json.dumps(per_law_db))

    cm = col.metadata or {}
    cfg_ok = cm.get("embedding_model") == MODELS[MODEL_KEY]["hf_id"] and cm.get("dim") == dim
    add("collection records the embedding model and dimension", cfg_ok, f"{cm.get('embedding_model')}, dim {cm.get('dim')}")
    hnsw = (col.configuration or {}).get("hnsw") or {}
    add("index uses cosine distance and the planned HNSW settings",
        all(hnsw.get(k) == v for k, v in HNSW.items()), json.dumps({k: hnsw.get(k) for k in HNSW}))
    stats = {"records": n, "ann_recall_at_10": round(recall, 4), "self_top1": f"{top1}/{len(ids)}", "per_law": per_law_db}
    return checks, stats


def print_checks(checks):
    for c in checks:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['check']}" + (f"  ({c['detail']})" if c["detail"] else ""))
    failed = sum(not c["ok"] for c in checks)
    print(f"\n{len(checks) - failed} checks passed, {failed} failed")
    return failed == 0


# ------------------------------------------------------------------ build
def build(args, chunks, make_embedder):
    client = get_client(args.db_dir)
    exists = args.collection in collection_names(client)
    if exists and not args.rebuild:
        sys.exit(f"Collection {args.collection!r} already exists in {args.db_dir}. "
                 f"Use --verify-only to check it or --rebuild to replace it.")

    cfg = MODELS[MODEL_KEY]
    texts = [passage_text(c, PASSAGE_FORMAT) for c in chunks]
    print(f"Encoding {len(texts)} chunks with {cfg['hf_id']} (max_length {MAX_LENGTH}, batch {BATCH_SIZE}) ...")
    emb_model = make_embedder(cfg, MAX_LENGTH, BATCH_SIZE)
    counts = emb_model.token_counts(texts)
    if max(counts) > MAX_LENGTH:
        sys.exit(f"A chunk has {max(counts)} tokens, more than max_length {MAX_LENGTH}; refusing to truncate.")
    vectors, secs = encode(emb_model, texts, "chunks")
    if hasattr(emb_model, "close"):
        emb_model.close()
    if vectors.shape != (len(chunks), cfg["dim"]) or not np.isfinite(vectors).all():
        sys.exit(f"Unexpected embedding matrix: {vectors.shape}")
    print(f"Encoded in {secs:.0f}s ({1000 * secs / len(chunks):.0f} ms per chunk)")

    # only now touch the database, so a failed encoding never destroys an existing index
    if exists:
        client.delete_collection(args.collection)
        print(f"Deleted existing collection {args.collection!r} (--rebuild)")
    col = client.create_collection(
        args.collection, embedding_function=None, configuration={"hnsw": HNSW},
        metadata={"description": "GermanyLex legal chunks", "embedding_model": cfg["hf_id"], "dim": cfg["dim"],
                  "pooling": cfg["pooling"], "normalized": True, "passage_format": PASSAGE_FORMAT,
                  "query_prefix": cfg["query_prefix"], "passage_prefix": cfg["passage_prefix"],
                  "max_length": MAX_LENGTH, "n_chunks": len(chunks), "built_on": time.strftime("%Y-%m-%d")})
    for s in range(0, len(chunks), ADD_BATCH):
        part = chunks[s:s + ADD_BATCH]
        col.add(ids=[c["chunk_id"] for c in part], embeddings=vectors[s:s + ADD_BATCH].tolist(),
                documents=[c["text"] for c in part], metadatas=[chunk_metadata(c) for c in part])
    print(f"Stored {col.count()} records in collection {args.collection!r}")
    return {"encode_seconds": round(secs, 1), "ms_per_chunk": round(1000 * secs / len(chunks), 1),
            "max_chunk_tokens": int(max(counts))}


def main(make_embedder=HFEmbedder, argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    ap.add_argument("--db-dir", type=Path, default=DB_DIR)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--expect-chunks", type=int, default=718)
    ap.add_argument("--report", type=Path, default=REPORT)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--rebuild", action="store_true", help="replace an existing collection")
    mode.add_argument("--verify-only", action="store_true", help="only re-open the database and verify it")
    args = ap.parse_args(argv)

    before = file_hashes(args.chunk_dir)
    chunks, sec_ids, _ = load_chunks(args.chunk_dir, args.expect_chunks)
    for c in chunks:
        chunk_metadata(c)                                            # fail early on bad metadata
    print(f"{len(chunks)} chunks | {len(sec_ids)} sections | database: {args.db_dir} | collection: {args.collection}")

    info = {}
    if not args.verify_only:
        info = build(args, chunks, make_embedder)

    print("\n=== Verification (new client, reading from disk) ===")
    checks, stats = verify(args.db_dir, args.collection, chunks)
    ok = print_checks(checks)
    unchanged = before == file_hashes(args.chunk_dir)
    print(f"Chunk files unchanged: {unchanged}")
    ok = ok and unchanged
    if args.db_dir.exists():
        print(f"Database size on disk: {dir_size_mb(args.db_dir)} MB")

    if not args.verify_only:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"script": "src/build_chroma_index.py", "built_on": time.strftime("%Y-%m-%d"), "collection": args.collection,
             "model": MODELS[MODEL_KEY]["hf_id"], "dim": MODELS[MODEL_KEY]["dim"], "distance": "cosine",
             "passage_format": PASSAGE_FORMAT, "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
             "n_chunks": len(chunks), "n_sections": len(sec_ids), "chunk_files_sha256": before,
             "chunk_files_unchanged": unchanged, "all_checks_passed": ok, "checks": checks, "stats": stats, **info},
            ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        print(f"Report written to {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
