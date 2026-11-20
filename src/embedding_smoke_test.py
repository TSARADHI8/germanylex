"""Small embedding smoke test: intfloat/multilingual-e5-base vs BAAI/bge-m3.

NOT the full embedding run. It embeds only a small pool of legal sections (the gold
sections of the DEV evaluation questions, their neighbouring sections as hard negatives,
and some random sections) plus the DEV questions, then checks
  1. that the vectors are valid (shape, finite, unit length, deterministic, padding-safe)
  2. whether each question ranks its gold section above the other sections in the pool

Dense embeddings only. No ChromaDB. Records and evaluation file are only read.

How each model turns text into one vector (done by hand here so it is visible):
  E5      tokens -> transformer -> MEAN of all token vectors (ignoring padding) -> L2 normalise
          inputs need prefixes:  "query: <question>"  and  "passage: <document>"
  BGE-M3  tokens -> transformer -> the FIRST token vector ([CLS]) -> L2 normalise
          no prefixes at all

Usage (repo root):
    python src/embedding_smoke_test.py
    python src/embedding_smoke_test.py --models e5          # one model only
    python src/embedding_smoke_test.py --passage-format text
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EVAL_FILE = ROOT / "data" / "eval" / "questions_v1.jsonl"
DEFAULT_OUT = ROOT / "reports" / "embedding_smoke_test.json"

# Specifications taken from the official model cards / Hugging Face file listings.
MODELS = {
    "e5": {
        "hf_id": "intfloat/multilingual-e5-base",
        "pooling": "mean", "query_prefix": "query: ", "passage_prefix": "passage: ",
        "dim": 768, "limit": 512, "params_m": 278, "languages": "94", "layers": 12,
    },
    "bge_m3": {
        "hf_id": "BAAI/bge-m3",
        "pooling": "cls", "query_prefix": "", "passage_prefix": "",
        "dim": 1024, "limit": 8192, "params_m": 568, "languages": "100+", "layers": 24,
    },
}


# ------------------------------------------------------------------ data
def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_data():
    records = []
    for p in sorted(PROCESSED.glob("*.jsonl")):
        records.extend(read_jsonl(p))
    by_id = {r["id"]: r for r in records}
    questions = [q for q in read_jsonl(EVAL_FILE) if q["split"] == "dev"]
    return records, by_id, questions


def build_pool(records, by_id, questions, n_random, seed):
    """Gold sections + their neighbours (hard negatives) + random distractors."""
    gold = {g["gold_id"] for q in questions for g in q["gold"]}
    primary = {g["gold_id"] for q in questions for g in q["gold"] if g["role"] == "primary"}
    pos = {r["id"]: i for i, r in enumerate(records)}
    pool = set(gold)
    for gid in primary:                      # previous/next section of the same law
        i = pos[gid]
        for j in (i - 1, i + 1):
            if 0 <= j < len(records) and records[j]["law"] == records[i]["law"]:
                pool.add(records[j]["id"])
    others = sorted(set(by_id) - pool)
    pool.update(random.Random(seed).sample(others, min(n_random, len(others))))
    return sorted(pool, key=lambda x: pos[x])


def passage_text(rec, fmt):
    return f'{rec["heading"]}\n{rec["text"]}' if fmt == "heading_text" else rec["text"]


# ------------------------------------------------------------------ embedders
class HFEmbedder:
    """Plain transformers implementation (no sentence-transformers needed)."""

    def __init__(self, cfg, max_length, batch_size):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.torch, self.cfg, self.max_length, self.batch_size = torch, cfg, max_length, batch_size
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(cfg["hf_id"], use_fast=True)
        self.model = AutoModel.from_pretrained(cfg["hf_id"], torch_dtype=torch.float32,
                                               low_cpu_mem_usage=True).eval()
        self.load_seconds = round(time.time() - t0, 1)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.weights_mb_fp32 = round(self.n_params * 4 / 1e6)
        self.hidden = self.model.config.hidden_size

    def token_counts(self, texts):
        enc = self.tok(texts, add_special_tokens=True, truncation=False)
        return [len(x) for x in enc["input_ids"]]

    def embed(self, texts):
        torch, F = self.torch, self.torch.nn.functional
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))   # less padding
        out = [None] * len(texts)
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = self.tok([texts[i] for i in idx], max_length=self.max_length,
                               truncation=True, padding=True, return_tensors="pt")
                h = self.model(**enc).last_hidden_state                       # (b, tokens, dim)
                if self.cfg["pooling"] == "mean":
                    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)             # mean over real tokens
                else:
                    v = h[:, 0]                                               # [CLS] token
                v = F.normalize(v, p=2, dim=1)
                for k, i in enumerate(idx):
                    out[i] = v[k].float().numpy()
        return np.stack(out)

    def close(self):
        del self.model, self.tok
        gc.collect()


def rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ checks and metrics
def validity_checks(emb, cfg, embedder, sample_texts):
    n, d = emb.shape
    norms = np.linalg.norm(emb, axis=1)
    sims = emb @ emb.T
    off = sims[~np.eye(n, dtype=bool)]
    # determinism and padding-invariance: re-embed 4 texts alone and in a different batch
    a = embedder.embed(sample_texts[:4])
    b = np.concatenate([embedder.embed([sample_texts[0]]), embedder.embed(sample_texts[1:4])])
    return {
        "shape": [int(n), int(d)],
        "expected_dim": cfg["dim"], "dim_matches_spec": int(d) == cfg["dim"],
        "dtype": str(emb.dtype),
        "all_finite": bool(np.isfinite(emb).all()),
        "norm_min": round(float(norms.min()), 6), "norm_max": round(float(norms.max()), 6),
        "unit_length_ok": bool(np.allclose(norms, 1.0, atol=1e-3)),
        "no_duplicate_vectors": bool((np.triu(sims, 1) < 0.99999).all()),
        "min_pairwise_cos": round(float(off.min()), 4),
        "mean_pairwise_cos": round(float(off.mean()), 4),
        "max_pairwise_cos": round(float(off.max()), 4),
        "batch_invariance_min_cos": round(float(np.min(np.sum(a * b, axis=1))), 6),
    }


def evaluate(q_emb, p_emb, questions, pool_ids, top_n=3):
    sims = q_emb @ p_emb.T
    pidx = {pid: i for i, pid in enumerate(pool_ids)}
    rows, hit = [], {1: 0, 3: 0, 5: 0}
    rr_sum = margin_sum = 0.0
    for qi, q in enumerate(questions):
        prim = [g["gold_id"] for g in q["gold"] if g["role"] == "primary"]
        allg = {g["gold_id"] for g in q["gold"]}
        order = np.argsort(-sims[qi])
        rank_of = {pool_ids[j]: r + 1 for r, j in enumerate(order)}
        best = min(rank_of[g] for g in prim)
        for k in hit:
            hit[k] += best <= k
        rr_sum += 1 / best
        best_gold = max(sims[qi, pidx[g]] for g in prim)
        best_other = max(sims[qi, j] for j, pid in enumerate(pool_ids) if pid not in allg)
        margin_sum += best_gold - best_other
        rows.append({
            "question_id": q["question_id"], "primary_gold": prim,
            "primary_gold_ranks": {g: rank_of[g] for g in prim},
            "best_primary_rank": best,
            "supporting_gold_ranks": {g["gold_id"]: rank_of[g["gold_id"]] for g in q["gold"] if g["role"] != "primary"},
            "gold_sim": round(float(best_gold), 4), "best_non_gold_sim": round(float(best_other), 4),
            "top3": [{"id": pool_ids[j], "sim": round(float(sims[qi, j]), 4), "is_gold": pool_ids[j] in allg}
                     for j in order[:top_n]],
        })
    n = len(questions)
    return {
        "n_questions": n, "pool_size": len(pool_ids),
        "hit_at_1": hit[1] / n, "hit_at_3": hit[3] / n, "hit_at_5": hit[5] / n,
        "mrr": round(rr_sum / n, 4), "mean_margin_gold_minus_best_other": round(margin_sum / n, 4),
        "similarity_range": {"min": round(float(sims.min()), 4), "mean": round(float(sims.mean()), 4),
                             "max": round(float(sims.max()), 4)},
        "per_question": rows,
    }


# ------------------------------------------------------------------ main
def run_model(name, cfg, args, questions, pool_ids, by_id, make_embedder):
    print(f"\n=== {name}: {cfg['hf_id']} ===")
    texts_p = [passage_text(by_id[i], args.passage_format) for i in pool_ids]
    texts_q = [q["question"] for q in questions]
    max_len = args.e5_max_length if name == "e5" else args.bge_max_length
    emb = make_embedder(cfg, max_len, args.batch_size)
    info = {"hf_id": cfg["hf_id"], "pooling": cfg["pooling"], "max_length_used": max_len,
            "documented_input_limit": cfg["limit"], "passage_format": args.passage_format,
            "load_seconds": getattr(emb, "load_seconds", None),
            "parameters": getattr(emb, "n_params", None),
            "weights_mb_fp32": getattr(emb, "weights_mb_fp32", None),
            "rss_mb_after_load": rss_mb()}
    p_in = [cfg["passage_prefix"] + t for t in texts_p]
    q_in = [cfg["query_prefix"] + t for t in texts_q]
    counts = emb.token_counts(p_in)
    info["passages_over_max_length"] = sum(c > max_len for c in counts)
    info["gold_sections_truncated"] = sorted(
        pid for pid, c in zip(pool_ids, counts)
        if c > max_len and pid in {g["gold_id"] for q in questions for g in q["gold"]})
    info["passage_tokens"] = {"min": min(counts), "median": int(np.median(counts)), "max": max(counts)}

    t0 = time.time()
    p_emb, q_emb = emb.embed(p_in), emb.embed(q_in)
    info["encode_seconds"] = round(time.time() - t0, 1)
    info["rss_mb_after_encode"] = rss_mb()
    result = {"info": info,
              "validity_passages": validity_checks(p_emb, cfg, emb, p_in),
              "validity_queries": validity_checks(q_emb, cfg, emb, q_in),
              "retrieval": evaluate(q_emb, p_emb, questions, pool_ids)}
    if name == "e5" and not args.no_ablation:            # what happens if the prefixes are forgotten
        ab = evaluate(emb.embed(texts_q), emb.embed(texts_p), questions, pool_ids)
        result["ablation_e5_without_prefixes"] = {k: ab[k] for k in
            ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "mean_margin_gold_minus_best_other", "similarity_range")}
    if hasattr(emb, "close"):
        emb.close()
    return result


def print_report(results, questions):
    print("\n================ SUMMARY ================")
    print(f"{'':8}{'dim':>6}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}{'margin':>9}{'trunc.':>8}{'enc s':>7}{'RAM MB':>8}")
    for name, r in results.items():
        v, e, i = r["validity_passages"], r["retrieval"], r["info"]
        print(f"{name:8}{v['shape'][1]:>6}{e['hit_at_1']:>8.2f}{e['hit_at_3']:>8.2f}{e['hit_at_5']:>8.2f}"
              f"{e['mrr']:>8.3f}{e['mean_margin_gold_minus_best_other']:>9.4f}{i['passages_over_max_length']:>8}"
              f"{i['encode_seconds']:>7}{str(i['rss_mb_after_encode']):>8}")
    for name, r in results.items():
        v = r["validity_passages"]
        print(f"\n[{name}] validity: dim ok={v['dim_matches_spec']} finite={v['all_finite']} "
              f"unit-length={v['unit_length_ok']} batch-invariance cos={v['batch_invariance_min_cos']} "
              f"duplicates-free={v['no_duplicate_vectors']} mean pairwise cos={v['mean_pairwise_cos']}")
        print(f"[{name}] similarity range: {r['retrieval']['similarity_range']}")
        if r["info"]["gold_sections_truncated"]:
            print(f"[{name}] gold sections cut by the length limit: {r['info']['gold_sections_truncated']}")
        if "ablation_e5_without_prefixes" in r:
            a = r["ablation_e5_without_prefixes"]
            print(f"[{name}] WITHOUT prefixes: hit@1={a['hit_at_1']:.2f} hit@3={a['hit_at_3']:.2f} MRR={a['mrr']}")
    names = list(results)
    print("\nPer question: rank of the best primary gold section (1 = best)")
    print(f"{'question':10}{'gold':26}" + "".join(f"{n:>9}" for n in names))
    for k, q in enumerate(questions):
        row = [results[n]["retrieval"]["per_question"][k] for n in names]
        print(f"{q['question_id']:10}{','.join(row[0]['primary_gold'])[:25]:26}" +
              "".join(f"{x['best_primary_rank']:>9}" for x in row))


def _to_builtin(o):
    """json fallback for numpy numbers/bools/arrays."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def main(make_embedder=HFEmbedder):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-random", type=int, default=20, help="random distractor sections in the pool")
    ap.add_argument("--passage-format", choices=["heading_text", "text"], default="heading_text")
    ap.add_argument("--e5-max-length", type=int, default=512)
    ap.add_argument("--bge-max-length", type=int, default=1024, help="BGE-M3 allows 8192; 1024 keeps RAM low")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    records, by_id, questions = load_data()
    pool_ids = build_pool(records, by_id, questions, args.n_random, args.seed)
    print(f"{len(records)} records read | {len(questions)} DEV questions | pool of {len(pool_ids)} sections")

    results = {}
    for name in args.models:                 # one model at a time to stay within 8 GB RAM
        results[name] = run_model(name, MODELS[name], args, questions, pool_ids, by_id, make_embedder)

    out = {"meta": {"script": "src/embedding_smoke_test.py", "seed": args.seed,
                    "split_used": "dev only", "n_questions": len(questions), "pool_ids": pool_ids,
                    "passage_format": args.passage_format,
                    "query_language_note": "most dev questions are English paraphrases of German posts; passages are German",
                    "specifications": {k: {kk: vv for kk, vv in v.items() if kk != "hf_id"} for k, v in MODELS.items()},
                    "caveat": "10 questions and a ~45-section pool: one question = 10 percentage points. Indicative only."},
           "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=_to_builtin)
    print_report(results, questions)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
