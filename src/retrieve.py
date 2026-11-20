"""Semantic retrieval over the persistent ChromaDB index (BGE-M3, 718 chunks).

    question -> BGE-M3 query embedding -> ChromaDB nearest chunks -> chunks with legal metadata

Each hit carries the chunk text, its legal metadata (law, section, heading, area, source URL,
paragraphs, chunk position) and both the cosine distance (Chroma's value) and the similarity
(1 - distance). The query is embedded exactly as in the evaluation: raw question text, no
prefix, same HFEmbedder, max_length and pooling (BGE-M3 CLS + L2 normalisation).

Reused: get_client, collection_names, DB_DIR, COLLECTION, MODEL_KEY, MAX_LENGTH, BATCH_SIZE
(build_chroma_index.py), HFEmbedder, MODELS, read_jsonl (embedding_smoke_test.py).
The index, the chunk files and the evaluation file are only read.

Usage (repo root):
    python src/retrieve.py "Can my spouse work while the Blue Card is pending?"
    python src/retrieve.py "Wie lange ist die Kündigungsfrist für Mieter?" "No hot water in my flat" -k 5
    python src/retrieve.py "..." --json                       # machine-readable output
    python src/retrieve.py "..." --sections 10                # chunks grouped into ranked legal sections
    python src/retrieve.py --verify-dev                       # check against the saved dev evaluation

Use from Python:
    from retrieve import Retriever
    hits = Retriever().retrieve("my question", k=10)          # list of dicts, best first
    sections = Retriever().retrieve_sections("my question", n=10)   # ranked sections (best chunk = section score)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_chroma_index import (BATCH_SIZE, COLLECTION, DB_DIR, MAX_LENGTH, MODEL_KEY,  # noqa: E402
                                collection_names, get_client)
from embedding_smoke_test import MODELS, HFEmbedder, read_jsonl  # noqa: E402

ROOT = HERE.parent
EVAL_FILE = ROOT / "data" / "eval" / "questions_v1.jsonl"
DEV_REPORT = ROOT / "reports" / "chunk_embedding_comparison.json"


class Retriever:
    def __init__(self, db_dir: Path = DB_DIR, collection: str = COLLECTION, make_embedder=HFEmbedder):
        client = get_client(db_dir)
        if collection not in collection_names(client):
            sys.exit(f"Collection {collection!r} not found in {db_dir}. Build it first: python src/build_chroma_index.py")
        self.col = client.get_collection(collection, embedding_function=None)
        cfg = MODELS[MODEL_KEY]
        meta = self.col.metadata or {}
        if meta.get("embedding_model") != cfg["hf_id"] or meta.get("dim") != cfg["dim"]:
            sys.exit(f"The index was built with {meta.get('embedding_model')} (dim {meta.get('dim')}), "
                     f"but this script embeds queries with {cfg['hf_id']} (dim {cfg['dim']}).")
        self.n_records = self.col.count()
        self.query_prefix = cfg["query_prefix"]                      # "" for BGE-M3
        self.embedder = make_embedder(cfg, MAX_LENGTH, BATCH_SIZE)

    def retrieve(self, question: str, k: int = 10) -> list[dict]:
        """Return the k most similar chunks, best first."""
        if not question or not question.strip():
            raise ValueError("The question is empty.")
        if not 1 <= k <= self.n_records:
            raise ValueError(f"k must be between 1 and {self.n_records}, got {k}.")
        vec = self.embedder.embed([self.query_prefix + question])[0]
        res = self.col.query(query_embeddings=[vec.tolist()], n_results=k,
                             include=["documents", "metadatas", "distances"])
        hits = []
        for rank, (cid, doc, meta, dist) in enumerate(
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1):
            hits.append({"rank": rank, "chunk_id": cid, "distance": round(float(dist), 6),
                         "similarity": round(1.0 - float(dist), 6),
                         "section_id": meta["section_id"], "law": meta["law"], "law_title": meta["law_title"],
                         "section": meta["section"], "heading": meta["heading"], "area": meta["area"],
                         "paragraphs": meta["paragraphs"], "chunk_index": meta["chunk_index"],
                         "n_chunks": meta["n_chunks"], "source_url": meta["source_url"],
                         "retrieval_date": meta["retrieval_date"], "text": doc})
        return hits

    def retrieve_sections(self, question: str, n: int = 20) -> list[dict]:
        """Top-n legal sections, best first. All chunks are ranked by ChromaDB (the index is small),
        grouped by section, and a section's score is its best chunk, exactly as in the evaluation.
        Ranking every chunk avoids the misses an approximate search can make with a small pool."""
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        return group_by_section(self.retrieve(question, k=self.n_records))[:n]


def best_per_section(hits: list[dict]) -> list[tuple[str, float]]:
    """Section score = its best chunk (the aggregation used in the evaluation); best section first."""
    best: dict[str, float] = {}
    for h in hits:
        best[h["section_id"]] = max(best.get(h["section_id"], -1.0), h["similarity"])
    return sorted(best.items(), key=lambda kv: -kv[1])


def group_by_section(hits: list[dict]) -> list[dict]:
    """Chunk hits (best first) -> sections ranked by their best chunk. Each section keeps the
    chunks that were retrieved for it (best first), so the grouping can be inspected."""
    secs: dict[str, dict] = {}
    for h in hits:
        s = secs.setdefault(h["section_id"], {
            "section_id": h["section_id"], "law": h["law"], "law_title": h["law_title"],
            "section": h["section"], "heading": h["heading"], "area": h["area"],
            "source_url": h["source_url"], "retrieval_date": h["retrieval_date"],
            "n_chunks_in_section": h["n_chunks"], "score": h["similarity"], "chunks": []})
        s["score"] = max(s["score"], h["similarity"])
        s["chunks"].append({"chunk_id": h["chunk_id"], "chunk_rank": h["rank"], "similarity": h["similarity"],
                            "paragraphs": h["paragraphs"], "text": h["text"]})
    ranked = sorted(secs.values(), key=lambda s: -s["score"])
    for i, s in enumerate(ranked, start=1):
        s["rank"] = i
    return ranked


def print_sections(question: str, sections: list[dict], full: bool):
    print(f"\nQuestion: {question}\n" + "-" * 78)
    for s in sections:
        print(f"#{s['rank']}  section score {s['score']:.4f}  (= best chunk)")
        print(f"    {s['law']} {s['section']}  {s['heading']}   [{s['area']}]")
        print(f"    {s['source_url']}")
        for c in s["chunks"]:
            print(f"    - chunk {c['chunk_id']}  similarity {c['similarity']:.4f}  (chunk rank {c['chunk_rank']})")
        best = s["chunks"][0]["text"]
        body = best if full else best.replace("\n", " ")[:220] + ("..." if len(best) > 220 else "")
        print("    " + body.replace("\n", "\n    "))


def print_hits(question: str, hits: list[dict], full: bool):
    print(f"\nQuestion: {question}\n" + "-" * 78)
    for h in hits:
        print(f"#{h['rank']}  similarity {h['similarity']:.4f}  (distance {h['distance']:.4f})")
        print(f"    {h['law']} {h['section']}  {h['heading']}   [{h['area']}]")
        print(f"    chunk {h['chunk_id']}  ({h['chunk_index']}/{h['n_chunks']}"
              + (f", paragraphs {h['paragraphs']}" if h["paragraphs"] else "") + ")")
        print(f"    {h['source_url']}")
        body = h["text"] if full else h["text"].replace("\n", " ")[:220] + ("..." if len(h["text"]) > 220 else "")
        print("    " + body.replace("\n", "\n    "))


def verify_dev(retriever: Retriever, eval_file: Path, report: Path, chunk_hits: int = 60, tol: float = 5e-4) -> bool:
    """Dev questions only: the top-3 sections from ChromaDB must equal the ones the evaluation
    computed by brute force (same order, scores within tol). Labels play no role in this check."""
    saved = {r["question_id"]: r for r in json.load(open(report, encoding="utf-8"))["results"]["bge_m3"]["per_question"]}
    dev = [q for q in read_jsonl(eval_file) if q["split"] == "dev"]
    if {q["question_id"] for q in dev} != set(saved):
        sys.exit("The questions in the saved dev report differ from the dev split of the evaluation file.")
    ok_all = True
    print(f"{'question':9}{'top-3 sections identical':26}{'max score diff':>16}")
    for q in dev:
        got = best_per_section(retriever.retrieve(q["question"], k=chunk_hits))[:3]
        want = [(t["id"], t["score"]) for t in saved[q["question_id"]]["top3_sections"]]
        same_ids = [g[0] for g in got] == [w[0] for w in want]
        diff = max(abs(g[1] - w[1]) for g, w in zip(got, want))
        ok = same_ids and diff <= tol
        ok_all &= ok
        print(f"{q['question_id']:9}{'yes' if same_ids else 'NO':26}{diff:>16.6f}   {'PASS' if ok else 'FAIL'}")
    print(f"\n{'All dev questions reproduced' if ok_all else 'MISMATCH'} (tolerance {tol}; dev split only)")
    return ok_all


def main(make_embedder=HFEmbedder, argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="*", help="one or more questions")
    ap.add_argument("-k", type=int, default=5, help="number of chunks to return (default 5)")
    ap.add_argument("--sections", type=int, metavar="N", help="group the chunks into the top-N ranked sections instead")
    ap.add_argument("--full", action="store_true", help="print the complete chunk text")
    ap.add_argument("--json", action="store_true", help="print the hits as JSON")
    ap.add_argument("--verify-dev", action="store_true", help="reproduce the saved dev evaluation from the index")
    ap.add_argument("--db-dir", type=Path, default=DB_DIR)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--eval-file", type=Path, default=EVAL_FILE)
    ap.add_argument("--dev-report", type=Path, default=DEV_REPORT)
    args = ap.parse_args(argv)
    if bool(args.questions) == args.verify_dev:
        ap.error("give one or more questions, or --verify-dev (not both)")

    r = Retriever(args.db_dir, args.collection, make_embedder)
    if args.verify_dev:
        return 0 if verify_dev(r, args.eval_file, args.dev_report) else 1
    out = {}
    for q in args.questions:
        if args.sections:
            hits = r.retrieve_sections(q, n=args.sections)
        else:
            hits = r.retrieve(q, k=args.k)
        out[q] = hits
        if not args.json:
            (print_sections if args.sections else print_hits)(q, hits, args.full)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
