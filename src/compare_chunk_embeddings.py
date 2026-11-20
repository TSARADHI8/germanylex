"""Chunk-level comparison: intfloat/multilingual-e5-base vs BAAI/bge-m3.

Default (--split dev): same 718 chunks, same development questions, same scoring for every
model. This is where the model was chosen.
Final check (--split test): BGE-M3 only, on the 29 test questions, run once with the settings
chosen on dev. Nothing is tuned on test (see the guards in main()).
NOT the final embedding run. No ChromaDB. The legal records, the chunk files and the
evaluation file are only read (their SHA256 is compared before and after the run).

How a question is scored (identical for every variant)
  1. embed every chunk ("heading\\ntext", the same string for every model) and every question
  2. cosine similarity question <-> each of the 718 chunks (vectors are unit length)
  3. section score = the BEST chunk of that section        (chunk -> section_id aggregation)
  4. rank the 398 sections by that score; the gold labels are section ids
  5. Hit@1 / Hit@3 / Hit@5 / MRR use the rank of the best PRIMARY gold section
     (same rule as the smoke test); a second line also counts supporting golds

Variants
  e5_prefix    E5 with "query: " / "passage: " (the documented way to use E5)
  e5_noprefix  E5 without prefixes (ablation)
  bge_m3       BGE-M3, dense vector only, no prefixes

Only the questions of the chosen --split are evaluated (default dev). The test split may only
be used with the single variant bge_m3, so it cannot be used to choose between models, and a
test run never overwrites existing files (dev reports included).

Reused from src/embedding_smoke_test.py: MODELS (specs, prefixes, pooling), HFEmbedder (manual
mean / CLS pooling + L2 normalisation), read_jsonl, rss_mb, _to_builtin.

Usage (repo root):
    python src/compare_chunk_embeddings.py
    python src/compare_chunk_embeddings.py --variants bge_m3 --output reports/bge_only.json
    python src/compare_chunk_embeddings.py --split test        # final check, BGE-M3 only, run once
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from embedding_smoke_test import MODELS, HFEmbedder, _to_builtin, read_jsonl, rss_mb  # noqa: E402

ROOT = HERE.parent
CHUNK_DIR = ROOT / "data" / "processed_chunks"
EVAL_FILE = ROOT / "data" / "eval" / "questions_v1.jsonl"
DEFAULT_PATHS = {   # (json, markdown) per split; the dev names are the ones used so far
    "dev": (ROOT / "reports" / "chunk_embedding_comparison.json", ROOT / "reports" / "chunk_embedding_comparison.md"),
    "test": (ROOT / "reports" / "chunk_embedding_comparison_test.json", ROOT / "reports" / "chunk_embedding_comparison_test.md"),
}
TEST_ALLOWED_VARIANTS = ["bge_m3"]

LAW_ORDER = ["arbzg", "aufenthg_2004", "beschv_2013", "bgb", "milog"]
VARIANTS = {
    "e5_prefix": {"model": "e5", "use_prefix": True, "label": "E5 with query:/passage: prefixes"},
    "e5_noprefix": {"model": "e5", "use_prefix": False, "label": "E5 without prefixes"},
    "bge_m3": {"model": "bge_m3", "use_prefix": True, "label": "BGE-M3 (dense, no prefixes)"},
}
REQUIRED = ("chunk_id", "section_id", "heading", "text")


# ------------------------------------------------------------------ data
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(chunk_dir: Path, eval_file: Path) -> dict:
    files = sorted(chunk_dir.glob("*.json*")) + [eval_file]
    return {str(p.name): sha256_of(p) for p in files}


def load_chunks(chunk_dir: Path, expect: int | None):
    chunks = []
    for law in LAW_ORDER:
        p = chunk_dir / f"{law}.jsonl"
        if not p.exists():
            sys.exit(f"Missing chunk file: {p}  (run src/chunk_documents.py first)")
        chunks.extend(read_jsonl(p))
    bad = [c.get("chunk_id") for c in chunks if any(not str(c.get(k, "")).strip() for k in REQUIRED if k != "heading")]
    if bad:
        sys.exit(f"Chunks with missing fields/empty text: {bad[:5]}")
    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        sys.exit("Duplicate chunk ids")
    if expect is not None and len(chunks) != expect:
        sys.exit(f"Expected {expect} chunks but found {len(chunks)}")
    sec_ids = list(dict.fromkeys(c["section_id"] for c in chunks))          # keeps file order
    pos = {s: i for i, s in enumerate(sec_ids)}
    members = [[] for _ in sec_ids]
    for j, c in enumerate(chunks):
        members[pos[c["section_id"]]].append(j)
    return chunks, sec_ids, [np.array(m) for m in members]


def load_questions(eval_file: Path, sec_ids, split: str):
    rows = read_jsonl(eval_file)
    dev = [q for q in rows if q["split"] == split]
    skipped = len(rows) - len(dev)
    known = set(sec_ids)
    for q in dev:
        if not q["gold"]:    # out-of-scope questions have no gold: retrieved and listed, not scored
            if q.get("scope_status") != "out_of_scope":
                sys.exit(f"{q['question_id']}: no gold but scope_status is {q.get('scope_status')!r}")
            continue
        missing = [g["gold_id"] for g in q["gold"] if g["gold_id"] not in known]
        if missing:
            sys.exit(f"{q['question_id']}: gold section(s) not in the chunk files: {missing}")
        if not any(g["role"] == "primary" for g in q["gold"]):
            sys.exit(f"{q['question_id']}: no primary gold")
    return dev, skipped


def passage_text(chunk, fmt):
    return f'{chunk["heading"]}\n{chunk["text"]}' if fmt == "heading_text" else chunk["text"]


# ------------------------------------------------------------------ measuring
class PeakRSS:
    """Samples the process RSS in a thread, so the peak is per phase (needs psutil)."""

    def __init__(self, interval=0.05):
        self.interval, self._stop, self._t, self.peak = interval, threading.Event(), None, None
        try:
            import psutil
            self.proc = psutil.Process()
        except Exception:  # noqa: BLE001
            self.proc = None

    def __enter__(self):
        if self.proc is not None:
            self.peak = self.proc.memory_info().rss

            def poll():
                while not self._stop.is_set():
                    self.peak = max(self.peak, self.proc.memory_info().rss)
                    time.sleep(self.interval)
            self._t = threading.Thread(target=poll, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        if self._t is not None:
            self._stop.set()
            self._t.join()
            self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def peak_mb(self):
        return None if self.peak is None else round(self.peak / 1e6)


def encode(emb, texts, label, block=64):
    """Embed in length-sorted blocks so progress can be shown; returns (vectors, seconds)."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out, t0 = [None] * len(texts), time.time()
    for s in range(0, len(order), block):
        idx = order[s:s + block]
        vecs = emb.embed([texts[i] for i in idx])
        for k, i in enumerate(idx):
            out[i] = vecs[k]
        print(f"    {label}: {min(s + block, len(order))}/{len(order)}  ({time.time() - t0:.0f}s)",
              end="\r", flush=True)
    print(" " * 70, end="\r")
    return np.stack(out), time.time() - t0


# ------------------------------------------------------------------ scoring
def evaluate(q_emb, c_emb, questions, chunks, sec_ids, members, top_n=3):
    sims = q_emb @ c_emb.T                                           # (questions, chunks)
    sec = np.stack([sims[:, m].max(axis=1) for m in members], axis=1)  # best chunk per section
    nq = sum(1 for q in questions if q["gold"])                      # scored questions (out-of-scope ones have no gold)
    hit, hit_any, rr, rr_any, margin = {1: 0, 3: 0, 5: 0}, {1: 0, 3: 0, 5: 0}, 0.0, 0.0, 0.0
    rows = []
    for qi, q in enumerate(questions):
        if not q["gold"]:                                            # out-of-scope: show what is retrieved, no metrics
            order = np.argsort(-sec[qi], kind="stable")
            rows.append({"question_id": q["question_id"], "out_of_scope": True, "primary_gold": [],
                         "best_primary_rank": None,
                         "top3_sections": [{"id": sec_ids[j], "score": round(float(sec[qi, j]), 4), "is_gold": False}
                                           for j in order[:top_n]]})
            continue
        prim = [g["gold_id"] for g in q["gold"] if g["role"] == "primary"]
        allg = {g["gold_id"] for g in q["gold"]}
        order = np.argsort(-sec[qi], kind="stable")
        rank_of = {sec_ids[j]: r + 1 for r, j in enumerate(order)}
        best = min(rank_of[g] for g in prim)
        best_any = min(rank_of[g] for g in allg)
        for k in hit:
            hit[k] += best <= k
            hit_any[k] += best_any <= k
        rr += 1 / best
        rr_any += 1 / best_any
        gold_sec = min(prim, key=lambda g: rank_of[g])               # best-ranked primary section
        m = members[sec_ids.index(gold_sec)]
        jc = int(m[np.argmax(sims[qi, m])])                          # its best chunk
        best_gold = float(sec[qi, sec_ids.index(gold_sec)])
        best_other = max(float(sec[qi, j]) for j, s in enumerate(sec_ids) if s not in allg)
        margin += best_gold - best_other
        rows.append({
            "question_id": q["question_id"], "primary_gold": prim,
            "primary_gold_ranks": {g: rank_of[g] for g in prim},
            "best_primary_rank": best, "best_any_gold_rank": best_any,
            "supporting_gold_ranks": {g["gold_id"]: rank_of[g["gold_id"]] for g in q["gold"] if g["role"] != "primary"},
            "gold_section_n_chunks": int(len(m)),
            "best_gold_chunk": chunks[jc]["chunk_id"],
            "best_gold_chunk_paragraphs": chunks[jc].get("paragraphs"),
            "best_gold_chunk_rank_among_all_chunks": int((sims[qi] > sims[qi, jc]).sum()) + 1,
            "gold_score": round(best_gold, 4), "best_non_gold_score": round(best_other, 4),
            "top3_sections": [{"id": sec_ids[j], "score": round(float(sec[qi, j]), 4), "is_gold": sec_ids[j] in allg}
                              for j in order[:top_n]],
        })
    return {
        "n_questions": nq, "n_out_of_scope_not_scored": len(questions) - nq, "n_sections": len(sec_ids), "n_chunks": len(chunks),
        "hit_at_1": hit[1] / nq, "hit_at_3": hit[3] / nq, "hit_at_5": hit[5] / nq, "mrr": round(rr / nq, 4),
        "any_gold_hit_at_1": hit_any[1] / nq, "any_gold_hit_at_3": hit_any[3] / nq,
        "any_gold_hit_at_5": hit_any[5] / nq, "any_gold_mrr": round(rr_any / nq, 4),
        "mean_margin_gold_minus_best_other": round(margin / nq, 4),
        "score_range": {"min": round(float(sims.min()), 4), "mean": round(float(sims.mean()), 4),
                        "max": round(float(sims.max()), 4)},
        "per_question": rows,
    }


def vector_checks(emb, dim):
    norms = np.linalg.norm(emb, axis=1)
    return {"shape": list(emb.shape), "dim_matches_spec": int(emb.shape[1]) == dim,
            "all_finite": bool(np.isfinite(emb).all()), "unit_length_ok": bool(np.allclose(norms, 1.0, atol=1e-3)),
            "no_duplicate_vectors": bool(len({v.tobytes() for v in emb}) == len(emb))}


# ------------------------------------------------------------------ run
def run_model(model_key, variant_keys, args, questions, chunks, sec_ids, members, make_embedder):
    cfg = MODELS[model_key]
    print(f"\n=== {model_key}: {cfg['hf_id']} ===")
    max_len = args.e5_max_length if model_key == "e5" else args.bge_max_length
    rss_before = rss_mb()
    with PeakRSS() as load_mem:
        emb = make_embedder(cfg, max_len, args.batch_size)
    base = {"hf_id": cfg["hf_id"], "pooling": cfg["pooling"], "dim": cfg["dim"], "max_length_used": max_len,
            "documented_input_limit": cfg["limit"], "load_seconds": getattr(emb, "load_seconds", None),
            "parameters": getattr(emb, "n_params", None), "weights_mb_fp32": getattr(emb, "weights_mb_fp32", None),
            "rss_mb_before_load": rss_before, "peak_rss_mb_during_load": load_mem.peak_mb}
    out = {}
    for vk in variant_keys:
        v = VARIANTS[vk]
        qp, pp = (cfg["query_prefix"], cfg["passage_prefix"]) if v["use_prefix"] else ("", "")
        c_in = [pp + passage_text(c, args.passage_format) for c in chunks]
        q_in = [qp + q["question"] for q in questions]
        counts = emb.token_counts(c_in)
        print(f"  [{vk}] query prefix={qp!r} passage prefix={pp!r} | chunk tokens max={max(counts)} "
              f"over limit {max_len}: {sum(c > max_len for c in counts)}")
        with PeakRSS() as mem:
            c_emb, t_c = encode(emb, c_in, "chunks")
            q_emb, t_q = encode(emb, q_in, "questions", block=len(q_in))
        res = evaluate(q_emb, c_emb, questions, chunks, sec_ids, members)
        res["info"] = dict(base, variant=vk, label=v["label"], query_prefix=qp, passage_prefix=pp,
                           passage_format=args.passage_format,
                           chunks_over_max_length=int(sum(c > max_len for c in counts)),
                           chunk_tokens={"min": min(counts), "median": int(np.median(counts)), "max": max(counts)},
                           encode_chunks_seconds=round(t_c, 1), encode_questions_seconds=round(t_q, 2),
                           ms_per_chunk=round(1000 * t_c / len(chunks), 1),
                           peak_rss_mb_during_encode=mem.peak_mb, rss_mb_after_encode=rss_mb())
        res["vector_checks_chunks"] = vector_checks(c_emb, cfg["dim"])
        res["vector_checks_questions"] = vector_checks(q_emb, cfg["dim"])
        out[vk] = res
        e = res
        print(f"  [{vk}] hit@1={e['hit_at_1']:.2f} hit@3={e['hit_at_3']:.2f} hit@5={e['hit_at_5']:.2f} "
              f"MRR={e['mrr']:.3f} | chunks {t_c:.0f}s, peak RSS {mem.peak_mb} MB")
    if hasattr(emb, "close"):
        emb.close()
    return out


def paired(results, a, b):
    ra = [r["best_primary_rank"] for r in results[a]["per_question"] if r["best_primary_rank"] is not None]
    rb = [r["best_primary_rank"] for r in results[b]["per_question"] if r["best_primary_rank"] is not None]
    return {"a": a, "b": b, "a_better": sum(x < y for x, y in zip(ra, rb)),
            "b_better": sum(y < x for x, y in zip(ra, rb)), "tie": sum(x == y for x, y in zip(ra, rb))}


def pct(x):
    return f"{100 * x:.0f}%"


def write_markdown(path, results, questions, meta, pairs):
    names = list(results)
    sp = meta["split"]
    title = ("# Chunk-level embedding comparison: E5 vs BGE-M3 (DEV split)" if sp == "dev" else
             "# Final evaluation: BGE-M3 on the TEST split (settings fixed on dev, run once)")
    L = [title, "",
         f"Generated by `src/compare_chunk_embeddings.py` on {meta['run_date']}.", "",
         "## Setup", "",
         f"- **Chunks:** {meta['n_chunks']} (from {meta['n_sections']} sections), identical for every variant. "
         f"Passage string: `{meta['passage_format']}`, the same for all models.",
         f"- **Questions:** {meta['n_scored']} scored {sp} questions"
         + (f" + {meta['n_out_of_scope']} out-of-scope (retrieved and listed below, not in the metrics)" if meta["n_out_of_scope"] else "")
         + f"; questions of the other split skipped: {meta['n_other_split_skipped']}.",
         "- **Scoring:** cosine similarity, section score = best chunk of the section, rank over all sections; "
         "hit/MRR use the best primary gold section.",
         "- **Truncation:** " + ("no chunk is cut for any variant." if not any(
             r["info"]["chunks_over_max_length"] for r in results.values()) else
             "; ".join(f"{r['info']['label']}: {r['info']['chunks_over_max_length']} chunk(s) over the input limit"
                       for r in results.values() if r["info"]["chunks_over_max_length"])), "",
         "## Retrieval (primary gold)", "",
         "| Variant | Hit@1 | Hit@3 | Hit@5 | MRR | margin | Hit@1 any gold | Hit@3 any gold | MRR any gold |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = results[n]
        L.append(f"| {r['info']['label']} | {pct(r['hit_at_1'])} | {pct(r['hit_at_3'])} | {pct(r['hit_at_5'])} | "
                 f"{r['mrr']:.3f} | {r['mean_margin_gold_minus_best_other']:+.4f} | {pct(r['any_gold_hit_at_1'])} | "
                 f"{pct(r['any_gold_hit_at_3'])} | {r['any_gold_mrr']:.3f} |")
    L += ["", "margin = best gold section score minus best non-gold section score (higher = clearer separation; "
              "only comparable within one model, cosine scales differ).", "",
          "## Cost", "",
          "| Variant | dim | params (M) | load s | chunks encode s | ms/chunk | questions s | peak RAM MB (encode) | chunks over limit |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        i = results[n]["info"]
        L.append(f"| {i['label']} | {i['dim']} | {round((i['parameters'] or 0) / 1e6)} | {i['load_seconds']} | "
                 f"{i['encode_chunks_seconds']} | {i['ms_per_chunk']} | {i['encode_questions_seconds']} | "
                 f"{i['peak_rss_mb_during_encode']} | {i['chunks_over_max_length']} |")
    L += ["", "Peak RAM is the whole Python process (model weights + activations + libraries) sampled during "
              "encoding. When several models run in one process it can include leftovers of the earlier model.", "",
          "## Rank of the best primary gold section per question (1 = best)", "",
          "| Question | Gold (primary) | " + " | ".join(names) + " | Question text |",
          "|---|---|" + "---|" * len(names) + "---|"]
    for k, q in enumerate(questions):
        rows = [results[n]["per_question"][k] for n in names]
        L.append(f"| {q['question_id']} | {', '.join(rows[0]['primary_gold']) or '(out of scope)'} | " +
                 " | ".join("-" if r["best_primary_rank"] is None else str(r["best_primary_rank"]) for r in rows)
                 + f" | {q['question'][:70]} |")
    oos = [(q, results[names[0]]["per_question"][k]) for k, q in enumerate(questions) if not q["gold"]]
    if oos:
        L += ["", f"## Out-of-scope questions (not scored): top-3 sections retrieved by {names[0]}", "",
              "| Question | Reason | Top-3 sections (cosine) |", "|---|---|---|"]
        for q, r in oos:
            L.append(f"| {q['question_id']} | {(q.get('out_of_scope_reason') or '')[:70]} | "
                     + ", ".join(f"{t['id']} ({t['score']:.3f})" for t in r["top3_sections"]) + " |")
    if pairs:
        L += ["", "## Paired comparison (questions where one variant ranks the gold section higher)", "",
              "| A | B | A better | B better | tie |", "|---|---|---|---|---|"]
        for p in pairs:
            L.append(f"| {p['a']} | {p['b']} | {p['a_better']} | {p['b_better']} | {p['tie']} |")
    L += ["", "## Reading the numbers", "",
          f"- With {meta['n_scored']} scored questions, one question moves a metric by {100 / meta['n_scored']:.0f} "
          "percentage points. Differences of one or two questions are noise.",
          f"- {meta['n_with_supporting']} of {meta['n_scored']} scored {sp} questions also have a supporting gold section; "
          "the 'any gold' columns show how much counting those changes the picture.",
          ("- This report used the dev split only. The test split is reserved for one confirmation of the chosen setup."
           if sp == "dev" else
           "- Test split, one run. The model and settings were chosen on dev; nothing here may be used to change them."), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")


def print_summary(results, questions, pairs, split):
    print(f"\n================ SUMMARY ({split} split, section-level, primary gold) ================")
    print(f"{'variant':13}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>8}{'margin':>9}{'enc s':>8}{'ms/chk':>8}{'RAM MB':>8}")
    for n, r in results.items():
        i = r["info"]
        print(f"{n:13}{r['hit_at_1']:>7.2f}{r['hit_at_3']:>7.2f}{r['hit_at_5']:>7.2f}{r['mrr']:>8.3f}"
              f"{r['mean_margin_gold_minus_best_other']:>9.4f}{i['encode_chunks_seconds']:>8}{i['ms_per_chunk']:>8}"
              f"{str(i['peak_rss_mb_during_encode']):>8}")
    print("\nRank of best primary gold section (1 = best):")
    print(f"{'question':10}{'gold':26}" + "".join(f"{n:>13}" for n in results))
    for k, q in enumerate(questions):
        rows = [results[n]["per_question"][k] for n in results]
        print(f"{q['question_id']:10}{(','.join(rows[0]['primary_gold']) or '(out of scope)')[:25]:26}" +
              "".join(f"{'-' if x['best_primary_rank'] is None else x['best_primary_rank']:>13}" for x in rows))
    for p in pairs:
        print(f"{p['a']} vs {p['b']}: {p['a']} better on {p['a_better']}, {p['b']} better on {p['b_better']}, tie {p['tie']}")


def main(make_embedder=HFEmbedder, argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev",
                    help="question split to evaluate; test allows only --variants bge_m3 (default: dev)")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None,
                    help="default: all variants for dev, bge_m3 for test")
    ap.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    ap.add_argument("--eval-file", type=Path, default=EVAL_FILE)
    ap.add_argument("--output", type=Path, default=None, help="default depends on --split")
    ap.add_argument("--report", type=Path, default=None, help="default depends on --split")
    ap.add_argument("--expect-chunks", type=int, default=718)
    ap.add_argument("--passage-format", choices=["heading_text", "text"], default="heading_text")
    ap.add_argument("--e5-max-length", type=int, default=512)
    ap.add_argument("--bge-max-length", type=int, default=1024, help="BGE-M3 allows 8192; chunks are <= ~460 tokens")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args(argv)

    # --- split guards: the test split is a one-shot check of the chosen model, never a model selection
    if args.variants is None:
        args.variants = TEST_ALLOWED_VARIANTS if args.split == "test" else list(VARIANTS)
    args.output = args.output or DEFAULT_PATHS[args.split][0]
    args.report = args.report or DEFAULT_PATHS[args.split][1]
    if args.split == "test":
        if args.variants != TEST_ALLOWED_VARIANTS:
            sys.exit(f"--split test allows only --variants {' '.join(TEST_ALLOWED_VARIANTS)}; "
                     f"the model is chosen on dev, not on test.")
        protected = {p.resolve() for p in DEFAULT_PATHS["dev"]}
        for p in (args.output, args.report):
            if p.resolve() in protected:
                sys.exit(f"Refusing to write the test results to a dev report file: {p}")
            if p.exists():
                sys.exit(f"{p} already exists. The test split is evaluated once; choose a new file name "
                         f"only if you really intend to repeat it.")
        print("*** TEST SPLIT: final one-shot evaluation of bge_m3 with the settings chosen on dev. ***")

    before = fingerprint(args.chunk_dir, args.eval_file)
    chunks, sec_ids, members = load_chunks(args.chunk_dir, args.expect_chunks)
    questions, n_skipped = load_questions(args.eval_file, sec_ids, args.split)
    other = "test" if args.split == "dev" else "dev"
    print(f"{len(chunks)} chunks | {len(sec_ids)} sections | {len(questions)} {args.split.upper()} questions "
          f"({n_skipped} {other} questions skipped, not used)")

    results = {}
    for model_key in dict.fromkeys(VARIANTS[v]["model"] for v in args.variants):   # load each model once
        keys = [v for v in args.variants if VARIANTS[v]["model"] == model_key]
        results.update(run_model(model_key, keys, args, questions, chunks, sec_ids, members, make_embedder))
    results = {v: results[v] for v in args.variants}

    pairs = []
    if "e5_prefix" in results and "bge_m3" in results:
        pairs.append(paired(results, "e5_prefix", "bge_m3"))
    if "e5_prefix" in results and "e5_noprefix" in results:
        pairs.append(paired(results, "e5_prefix", "e5_noprefix"))

    after = fingerprint(args.chunk_dir, args.eval_file)
    unchanged = before == after
    n_scored = sum(1 for q in questions if q["gold"])
    meta = {"script": "src/compare_chunk_embeddings.py", "run_date": time.strftime("%Y-%m-%d"),
            "split": args.split, "split_used": f"{args.split} only", "n_questions": len(questions),
            "n_scored": sum(1 for q in questions if q["gold"]), "n_out_of_scope": sum(1 for q in questions if not q["gold"]),
            "question_ids": [q["question_id"] for q in questions],
            "n_other_split_skipped": n_skipped,
            "n_with_supporting": sum(any(g["role"] != "primary" for g in q["gold"]) for q in questions), "n_chunks": len(chunks), "n_sections": len(sec_ids),
            "passage_format": args.passage_format, "aggregation": "section score = max cosine over its chunks",
            "input_sha256": before, "inputs_unchanged": unchanged,
            "caveat": f"{n_scored} scored {args.split} questions: one question = {100 / n_scored:.1f} percentage points. Indicative only."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"meta": meta, "pairs": pairs, "results": results}, fh, ensure_ascii=False, indent=2,
                  default=_to_builtin)
    write_markdown(args.report, results, questions, meta, pairs)
    print_summary(results, questions, pairs, args.split)
    print(f"\nInput chunk files and evaluation file unchanged: {unchanged}")
    print(f"Report written to {args.report}\nJSON written to  {args.output}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
