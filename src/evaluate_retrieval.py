"""Section-level retrieval evaluation through the ChromaDB index (BGE-M3): Hit@1/3/5/10/20 and MRR.

    question -> Retriever.retrieve_sections (ChromaDB chunks -> sections ranked by best chunk)
             -> rank of the gold section(s) -> Hit@k, MRR

Same protocol as src/compare_chunk_embeddings.py: section score = best chunk, rank over all
398 sections, hit/MRR use the best-ranked primary gold section ("any gold" also counts supporting
sections). The ranking here comes from the persistent ChromaDB index instead of in-memory
brute force; each gold rank is compared with the saved brute-force report of the same split.
Out-of-scope questions (no gold) are retrieved and listed but not scored.

--split dev (default) is the development set. --split test re-measures the held-out set with the
same unchanged model, index and code. It exists for reporting only: the script has no tunable
settings, and nothing may be changed because of the test numbers.

Existing reports are never overwritten: the default output files are new
(reports/section_retrieval_<split>.json/.md) and a run refuses to write over any existing file.
The index, the chunk files and the evaluation file are only read (SHA256 checked before and after).

Usage (repo root):
    python src/evaluate_retrieval.py                  # dev
    python src/evaluate_retrieval.py --split test     # held-out set, reporting only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_chroma_index import COLLECTION, DB_DIR  # noqa: E402
from compare_chunk_embeddings import CHUNK_DIR, EVAL_FILE, fingerprint, load_chunks, load_questions  # noqa: E402
from embedding_smoke_test import HFEmbedder  # noqa: E402
from retrieve import Retriever  # noqa: E402

ROOT = HERE.parent
KS = (1, 3, 5, 10, 20)
SAVED_REPORT = {"dev": ROOT / "reports" / "chunk_embedding_comparison.json",
                "test": ROOT / "reports" / "chunk_embedding_comparison_test.json"}


def out_paths(split):
    return (ROOT / "reports" / f"section_retrieval_{split}.json", ROOT / "reports" / f"section_retrieval_{split}.md")


def short(q, n=70):
    t = " ".join(q.split())
    return t if len(t) <= n else t[:n - 1] + "…"


def label(s):
    return f"{s['law']} {s['section']} {s['heading']}"


# ------------------------------------------------------------------ evaluation
def evaluate(retriever, questions, n_sections):
    rows = []
    for i, q in enumerate(questions, start=1):
        secs = retriever.retrieve_sections(q["question"], n=n_sections)         # full ranking of all sections
        if len(secs) != n_sections:
            sys.exit(f"{q['question_id']}: ChromaDB returned {len(secs)} sections instead of {n_sections}.")
        by_id = {s["section_id"]: s for s in secs}
        top = [{"rank": s["rank"], "section_id": s["section_id"], "label": label(s), "score": round(s["score"], 4),
                "best_chunk": s["chunks"][0]["chunk_id"]} for s in secs[:5]]
        print(f"  {i}/{len(questions)} {q['question_id']}", end="\r", flush=True)
        if not q["gold"]:                                                       # out-of-scope: no metrics
            rows.append({"question_id": q["question_id"], "question": q["question"], "out_of_scope": True,
                         "top_sections": top[:3]})
            continue
        prim = [g["gold_id"] for g in q["gold"] if g["role"] == "primary"]
        allg = [g["gold_id"] for g in q["gold"]]
        ranks = {g: by_id[g]["rank"] for g in allg}
        rows.append({
            "question_id": q["question_id"], "question": q["question"], "primary_gold": prim,
            "gold_ranks": ranks, "best_primary_rank": min(ranks[g] for g in prim),
            "best_any_gold_rank": min(ranks.values()),
            "gold_detail": [{"id": g, "label": label(by_id[g]), "role": next(x["role"] for x in q["gold"] if x["gold_id"] == g),
                             "rank": ranks[g], "score": round(by_id[g]["score"], 4),
                             "n_chunks_in_section": by_id[g]["n_chunks_in_section"]} for g in allg],
            "top_sections": top,
        })
    print(" " * 40, end="\r")
    return rows


def metrics(rows, ks):
    sc = [r for r in rows if not r.get("out_of_scope")]
    n = len(sc)
    prim = [r["best_primary_rank"] for r in sc]
    anyg = [r["best_any_gold_rank"] for r in sc]
    return {
        "n_scored": n, "n_out_of_scope_not_scored": len(rows) - n,
        "hit_at": {str(k): sum(r <= k for r in prim) for k in ks},
        "hit_rate_at": {str(k): round(sum(r <= k for r in prim) / n, 4) for k in ks},
        "any_gold_hit_at": {str(k): sum(r <= k for r in anyg) for k in ks},
        "any_gold_hit_rate_at": {str(k): round(sum(r <= k for r in anyg) / n, 4) for k in ks},
        "mrr": round(sum(1 / r for r in prim) / n, 4), "any_gold_mrr": round(sum(1 / r for r in anyg) / n, 4),
        "median_primary_rank": statistics.median(prim), "worst_primary_rank": max(prim),
    }


def consistency(rows, saved_path):
    """Compare every gold rank with the saved brute-force report of the same split."""
    if not saved_path.exists():
        return {"available": False, "note": f"{saved_path.name} not found"}
    saved = {r["question_id"]: r for r in json.load(open(saved_path, encoding="utf-8"))["results"]["bge_m3"]["per_question"]
             if not r.get("out_of_scope")}
    same, total, diffs = 0, 0, []
    for r in rows:
        if r.get("out_of_scope"):
            continue
        s = saved.get(r["question_id"])
        if s is None:
            diffs.append({"question_id": r["question_id"], "problem": "not in the saved report"})
            continue
        want = {**s["primary_gold_ranks"], **s["supporting_gold_ranks"]}     # section id -> rank (brute force)
        for g, rk in r["gold_ranks"].items():
            total += 1
            if want.get(g) == rk:
                same += 1
            else:
                diffs.append({"question_id": r["question_id"], "section": g, "chroma_rank": rk, "saved_rank": want.get(g)})
    return {"available": True, "report": saved_path.name, "gold_ranks_compared": total,
            "identical": same, "differences": diffs}


# ------------------------------------------------------------------ output
def pct(x):
    return f"{100 * x:.0f}%"


def write_markdown(path, split, rows, m, cons, meta):
    L = [f"# Section-level retrieval through ChromaDB: BGE-M3 ({split.upper()} split)", "",
         f"Generated by `src/evaluate_retrieval.py` on {meta['run_date']}.", "", "## Setup", "",
         f"- **Index:** collection `{meta['collection']}`, {meta['n_chunks']} chunks from {meta['n_sections']} sections, BAAI/bge-m3 (dense, 1024 dim, no prefixes).",
         "- **Chunks to sections:** the question is embedded, the nearest chunks are fetched from ChromaDB and grouped by section; a section's score is its best chunk; sections are ranked over all sections.",
         f"- **Questions:** {m['n_scored']} scored from the {split} split" + (f"; {m['n_out_of_scope_not_scored']} out-of-scope questions are listed but not scored." if m["n_out_of_scope_not_scored"] else "."),
         "- **Gold:** hit/MRR use the best-ranked primary section; 'any gold' also counts supporting sections. Labels: `data/eval/questions_v1.jsonl` as committed (q013 corrected).", ""]
    if split == "test":
        L += ["> Held-out set: re-measurement of the model, index and code chosen on dev. Reporting only; nothing was or may be tuned on these numbers.", ""]
    L += ["## Hit@k and MRR", "", "| | " + " | ".join(f"Hit@{k}" for k in KS) + " | MRR |", "|---|" + "---|" * (len(KS) + 1),
          "| Primary gold | " + " | ".join(f"{m['hit_at'][str(k)]}/{m['n_scored']} ({pct(m['hit_rate_at'][str(k)])})" for k in KS) + f" | {m['mrr']:.3f} |",
          "| Any gold | " + " | ".join(f"{m['any_gold_hit_at'][str(k)]}/{m['n_scored']} ({pct(m['any_gold_hit_rate_at'][str(k)])})" for k in KS) + f" | {m['any_gold_mrr']:.3f} |",
          "", f"Median rank of the primary gold section: {m['median_primary_rank']}; worst: {m['worst_primary_rank']} (of {meta['n_sections']} sections).", "",
          "## Rank per question (1 = best)", "", "| Question | Primary gold | Rank | Best any-gold rank | Top-1 retrieved | Question text |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("out_of_scope"):
            continue
        t1 = r["top_sections"][0]
        L.append(f"| {r['question_id']} | {', '.join(r['primary_gold'])} | {r['best_primary_rank']} | {r['best_any_gold_rank']} | {t1['label']} ({t1['score']:.3f}) | {short(r['question'])} |")
    fails = [r for r in rows if not r.get("out_of_scope") and r["best_primary_rank"] > 5]
    L += ["", f"## Misses: primary gold section not in the top 5 ({len(fails)})", ""]
    for r in fails:
        L.append(f"**{r['question_id']}** — {short(r['question'], 110)}")
        for g in r["gold_detail"]:
            L.append(f"- gold ({g['role']}): {g['label']} → rank {g['rank']} (score {g['score']:.3f})")
        L.append("- retrieved: " + "; ".join(f"#{t['rank']} {t['label']} ({t['score']:.3f})" for t in r["top_sections"][:3]))
        L.append("")
    oos = [r for r in rows if r.get("out_of_scope")]
    if oos:
        L += ["## Out-of-scope questions (not scored)", "", "| Question | Top-1 retrieved | Score | Question text |", "|---|---|---|---|"]
        for r in oos:
            t1 = r["top_sections"][0]
            L.append(f"| {r['question_id']} | {t1['label']} | {t1['score']:.3f} | {short(r['question'])} |")
        L += ["", "Listed for information only. No abstention threshold is derived from them.", ""]
    L += ["## Consistency with the brute-force evaluation", ""]
    if cons["available"]:
        L.append(f"{cons['identical']}/{cons['gold_ranks_compared']} gold ranks are identical to `{cons['report']}` (in-memory cosine ranking of the same 718 chunk vectors).")
        for d in cons["differences"]:
            L.append(f"- difference: {d}")
    else:
        L.append(f"Not checked: {cons['note']}.")
    L += ["", f"Inputs unchanged (SHA256 before/after): {meta['inputs_unchanged']}.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def print_summary(split, rows, m, cons):
    print(f"\n=== Section-level retrieval, {split.upper()} split ({m['n_scored']} scored, {m['n_out_of_scope_not_scored']} out-of-scope not scored) ===")
    print(f"{'':14}" + "".join(f"Hit@{k:<7}" for k in KS) + "MRR")
    print(f"{'primary gold':14}" + "".join(f"{m['hit_at'][str(k)]:>2}/{m['n_scored']} {pct(m['hit_rate_at'][str(k)]):<4}" + " " for k in KS) + f"{m['mrr']:.3f}")
    print(f"{'any gold':14}" + "".join(f"{m['any_gold_hit_at'][str(k)]:>2}/{m['n_scored']} {pct(m['any_gold_hit_rate_at'][str(k)]):<4}" + " " for k in KS) + f"{m['any_gold_mrr']:.3f}")
    print(f"median primary rank {m['median_primary_rank']}, worst {m['worst_primary_rank']}\n")
    print(f"{'question':9}{'primary gold':26}{'rank':>5}{'any':>5}  top-1 retrieved")
    for r in rows:
        if r.get("out_of_scope"):
            print(f"{r['question_id']:9}{'(out of scope)':26}{'-':>5}{'-':>5}  {r['top_sections'][0]['label']} ({r['top_sections'][0]['score']:.3f})")
        else:
            print(f"{r['question_id']:9}{', '.join(r['primary_gold']):26}{r['best_primary_rank']:>5}{r['best_any_gold_rank']:>5}  {r['top_sections'][0]['label']}")
    if cons["available"]:
        print(f"\nConsistency with {cons['report']}: {cons['identical']}/{cons['gold_ranks_compared']} gold ranks identical")
        for d in cons["differences"]:
            print("  difference:", d)
    else:
        print(f"\nConsistency not checked: {cons['note']}")


def main(make_embedder=HFEmbedder, argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--output", type=Path, help="JSON path (default reports/section_retrieval_<split>.json)")
    ap.add_argument("--report", type=Path, help="Markdown path (default reports/section_retrieval_<split>.md)")
    ap.add_argument("--db-dir", type=Path, default=DB_DIR)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    ap.add_argument("--eval-file", type=Path, default=EVAL_FILE)
    ap.add_argument("--expect-chunks", type=int, default=718)
    args = ap.parse_args(argv)
    d_json, d_md = out_paths(args.split)
    out_json, out_md = args.output or d_json, args.report or d_md
    protected = {p.resolve() for p in SAVED_REPORT.values()}
    for p in (out_json, out_md):
        if p.resolve() in protected or p.exists():
            sys.exit(f"Refusing to write {p}: it already exists (earlier reports are never overwritten). Choose another path.")

    before = fingerprint(args.chunk_dir, args.eval_file)
    chunks, sec_ids, _ = load_chunks(args.chunk_dir, args.expect_chunks)
    questions, skipped = load_questions(args.eval_file, sec_ids, args.split)
    print(f"{len(chunks)} chunks | {len(sec_ids)} sections | {args.split} questions: {len(questions)} (other split skipped: {skipped})")
    retriever = Retriever(args.db_dir, args.collection, make_embedder)
    if retriever.n_records != len(chunks):
        sys.exit(f"The index holds {retriever.n_records} records but the chunk files hold {len(chunks)}.")

    t0 = time.time()
    rows = evaluate(retriever, questions, len(sec_ids))
    m = metrics(rows, KS)
    cons = consistency(rows, SAVED_REPORT[args.split])
    after = fingerprint(args.chunk_dir, args.eval_file)
    meta = {"script": "src/evaluate_retrieval.py", "run_date": date.today().isoformat(), "split": args.split,
            "model": "BAAI/bge-m3", "collection": args.collection, "n_chunks": len(chunks), "n_sections": len(sec_ids),
            "ks": list(KS), "aggregation": "section score = best chunk similarity; rank over all sections",
            "retrieval_seconds": round(time.time() - t0, 1), "input_sha256": after, "inputs_unchanged": before == after,
            "note": ("Held-out set re-measured with unchanged model, index and code. Reporting only, no tuning."
                     if args.split == "test" else "Development set.")}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "x", encoding="utf-8") as fh:                         # "x": never overwrite
        json.dump({"meta": meta, "metrics": m, "consistency": cons, "per_question": rows}, fh, ensure_ascii=False, indent=2)
    write_markdown(out_md, args.split, rows, m, cons, meta)
    print_summary(args.split, rows, m, cons)
    print(f"\nInputs unchanged: {meta['inputs_unchanged']}\nReports written to {out_json}\n               and {out_md}")
    return 0 if meta["inputs_unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
