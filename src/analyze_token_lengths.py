"""Token-length analysis of the GermanyLex legal records.

Measures how many tokens every stored legal section has under the REAL tokenizers of
two candidate embedding models, so we can judge whether whole-section chunking fits
each model's input limit.

    intfloat/multilingual-e5-base   limit 512 tokens, documents need the "passage: " prefix
    BAAI/bge-m3                     limit 8192 tokens, no prefix

Only tokenizer files are downloaded (tokenizer.json, sentencepiece model, small configs).
No model weights are downloaded and no embeddings are computed.
Records and the evaluation dataset are only read, never modified.

Usage (from the repo root):
    python src/analyze_token_lengths.py
    python src/analyze_token_lengths.py --output reports/token_length_analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "token_length_analysis.json"
EXPECTED_RECORDS = 398

# name -> (hugging face id, text prefix, documented input limit in tokens)
MODELS = {
    "e5": {
        "hf_id": "intfloat/multilingual-e5-base",
        "prefix": "passage: ",
        "documented_limit": 512,
    },
    "bge_m3": {
        "hf_id": "BAAI/bge-m3",
        "prefix": "",
        "documented_limit": 8192,
    },
}
EXTRA_THRESHOLDS = [128, 256, 384, 512, 1024, 2048]
BATCH_SIZE = 32


# --------------------------------------------------------------------------- helpers
def percentile(sorted_values: list[int], p: float) -> float:
    """Linear-interpolation percentile (same convention as numpy's default)."""
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(sorted_values[int(k)])
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def summarize(values: list[int]) -> dict:
    s = sorted(values)
    return {
        "count": len(s),
        "min": s[0],
        "max": s[-1],
        "mean": round(statistics.fmean(s), 2),
        "median": round(statistics.median(s), 2),
        "p90": round(percentile(s, 90), 2),
        "p95": round(percentile(s, 95), 2),
        "p99": round(percentile(s, 99), 2),
    }


def load_records() -> list[dict]:
    files = sorted(glob.glob(str(PROCESSED_DIR / "*.jsonl")))
    if not files:
        sys.exit(f"No .jsonl files found in {PROCESSED_DIR}")
    records = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        sys.exit("Duplicate record IDs found; aborting.")
    if len(records) != EXPECTED_RECORDS:
        print(f"WARNING: expected {EXPECTED_RECORDS} records but found {len(records)}.")
    return records


def load_tokenizer(hf_id: str):
    """Loads the tokenizer only. AutoTokenizer fetches tokenizer files, never model weights."""
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(hf_id, use_fast=True)


def count_tokens(tokenizer, texts: list[str]) -> list[int]:
    """Token count per text, including the special tokens the model really receives."""
    lengths: list[int] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(batch, add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


def effective_limit(tokenizer, documented: int) -> dict:
    """Documented limit, plus what the tokenizer itself reports (sanity check)."""
    reported = getattr(tokenizer, "model_max_length", None)
    sane = isinstance(reported, int) and 0 < reported < 10**6
    return {
        "used": documented,
        "tokenizer_model_max_length": reported if sane else None,
        "matches_tokenizer": (reported == documented) if sane else None,
    }


def over_limit_stats(lengths: list[int], limit: int) -> dict:
    over = [n for n in lengths if n > limit]
    return {
        "limit": limit,
        "records_over_limit": len(over),
        "share_over_limit_pct": round(100 * len(over) / len(lengths), 2),
        "min_chunks_needed_extra": sum(math.ceil(n / limit) - 1 for n in over),
    }


def threshold_counts(lengths: list[int]) -> dict:
    return {f">{t}": sum(1 for n in lengths if n > t) for t in EXTRA_THRESHOLDS}


def per_law(records, e5_len, bge_len, e5_limit, bge_limit) -> dict:
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["law"]].append(i)
    out = {}
    for law, idx in sorted(groups.items()):
        out[law] = {
            "records": len(idx),
            "e5": {**summarize([e5_len[i] for i in idx]),
                   "over_limit": sum(e5_len[i] > e5_limit for i in idx)},
            "bge_m3": {**summarize([bge_len[i] for i in idx]),
                       "over_limit": sum(bge_len[i] > bge_limit for i in idx)},
        }
    return out


def comparison(e5_len: list[int], bge_len: list[int]) -> dict:
    diffs = [a - b for a, b in zip(e5_len, bge_len)]
    ratios = [a / b for a, b in zip(e5_len, bge_len)]
    return {
        "note": "e5 counts include the 'passage: ' prefix; bge_m3 counts have no prefix.",
        "diff_e5_minus_bge_m3": summarize_signed(diffs),
        "ratio_e5_over_bge_m3": {
            "mean": round(statistics.fmean(ratios), 4),
            "median": round(statistics.median(ratios), 4),
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
        },
        "records_e5_longer": sum(d > 0 for d in diffs),
        "records_equal": sum(d == 0 for d in diffs),
        "records_bge_m3_longer": sum(d < 0 for d in diffs),
        "pearson_correlation": round(pearson(e5_len, bge_len), 5),
    }


def summarize_signed(values: list[int]) -> dict:
    s = sorted(values)
    return {"min": s[0], "max": s[-1], "mean": round(statistics.fmean(s), 2),
            "median": round(statistics.median(s), 2)}


def pearson(x: list[int], y: list[int]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def top_longest(records, e5_len, bge_len, key_len, e5_limit, bge_limit, n=10) -> list[dict]:
    order = sorted(range(len(records)), key=lambda i: key_len[i], reverse=True)[:n]
    rows = []
    for rank, i in enumerate(order, 1):
        r = records[i]
        rows.append({
            "rank": rank,
            "id": r["id"],
            "law": r["law"],
            "section": r["section"],
            "heading": r.get("heading"),
            "characters": len(r["text"]),
            "e5_tokens_with_prefix": e5_len[i],
            "bge_m3_tokens": bge_len[i],
            "exceeds_e5_limit": e5_len[i] > e5_limit,
            "exceeds_bge_m3_limit": bge_len[i] > bge_limit,
        })
    return rows


def analyse(records: list[dict]) -> dict:
    texts = [r["text"] for r in records]
    # Secondary view: what we would embed if we also add law/section/heading as context.
    with_context = [f'{r["law"]} {r["section"]} {r.get("heading") or ""}\n{r["text"]}' for r in records]

    tokenizers, limits = {}, {}
    lengths = {}          # model -> per-record token counts (text only)
    lengths_ctx = {}      # model -> per-record token counts (text + heading context)
    for name, cfg in MODELS.items():
        print(f"Loading tokenizer: {cfg['hf_id']} (tokenizer files only) ...")
        tok = load_tokenizer(cfg["hf_id"])
        tokenizers[name] = tok
        limits[name] = effective_limit(tok, cfg["documented_limit"])
        lengths[name] = count_tokens(tok, [cfg["prefix"] + t for t in texts])
        lengths_ctx[name] = count_tokens(tok, [cfg["prefix"] + t for t in with_context])
        print(f"  counted {len(records)} records for {name}")

    e5_len, bge_len = lengths["e5"], lengths["bge_m3"]
    e5_lim, bge_lim = limits["e5"]["used"], limits["bge_m3"]["used"]

    result = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "src/analyze_token_lengths.py",
            "records_analysed": len(records),
            "record_files": sorted(os.path.basename(p) for p in glob.glob(str(PROCESSED_DIR / "*.jsonl"))),
            "python": sys.version.split()[0],
            "library_versions": library_versions(),
            "weights_downloaded": False,
            "embeddings_computed": False,
            "text_measured": "record['text'] only; special tokens included; no truncation",
            "models": {
                name: {
                    "hf_id": cfg["hf_id"],
                    "text_prefix": cfg["prefix"],
                    "input_limit": limits[name],
                    "tokenizer_class": type(tokenizers[name]).__name__,
                }
                for name, cfg in MODELS.items()
            },
        },
        "token_stats": {
            "e5_with_passage_prefix": summarize(e5_len),
            "bge_m3": summarize(bge_len),
        },
        "over_limit": {
            "e5": over_limit_stats(e5_len, e5_lim),
            "bge_m3": over_limit_stats(bge_len, bge_lim),
        },
        "records_above_thresholds": {
            "e5": threshold_counts(e5_len),
            "bge_m3": threshold_counts(bge_len),
        },
        "comparison": comparison(e5_len, bge_len),
        "per_law": per_law(records, e5_len, bge_len, e5_lim, bge_lim),
        "top10_longest_by_e5": top_longest(records, e5_len, bge_len, e5_len, e5_lim, bge_lim),
        "top10_longest_by_bge_m3": top_longest(records, e5_len, bge_len, bge_len, e5_lim, bge_lim),
        "secondary_with_heading_context": {
            "note": "Same measurement when 'LAW § section heading' is prepended to the text.",
            "e5_with_passage_prefix": {**summarize(lengths_ctx["e5"]),
                                       "over_limit": sum(n > e5_lim for n in lengths_ctx["e5"])},
            "bge_m3": {**summarize(lengths_ctx["bge_m3"]),
                       "over_limit": sum(n > bge_lim for n in lengths_ctx["bge_m3"])},
        },
        "records": [
            {"id": r["id"], "law": r["law"], "section": r["section"],
             "characters": len(r["text"]),
             "e5_tokens_with_prefix": e5_len[i], "bge_m3_tokens": bge_len[i]}
            for i, r in enumerate(records)
        ],
    }
    return result


def library_versions() -> dict:
    out = {}
    for mod in ("transformers", "tokenizers", "huggingface_hub"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def print_summary(res: dict) -> None:
    e5, bge = res["token_stats"]["e5_with_passage_prefix"], res["token_stats"]["bge_m3"]
    print("\n=== Token length summary (398 legal sections) ===")
    print(f"{'':10}{'min':>6}{'max':>7}{'mean':>9}{'median':>9}{'p90':>8}{'p95':>8}{'p99':>8}")
    for label, s in (("E5", e5), ("BGE-M3", bge)):
        print(f"{label:10}{s['min']:>6}{s['max']:>7}{s['mean']:>9}{s['median']:>9}{s['p90']:>8}{s['p95']:>8}{s['p99']:>8}")
    for name, label in (("e5", "E5 (limit 512)"), ("bge_m3", "BGE-M3 (limit 8192)")):
        o = res["over_limit"][name]
        print(f"{label}: {o['records_over_limit']} records over the limit ({o['share_over_limit_pct']}%)")
    print("\nTop 10 longest (by E5 tokens):")
    for row in res["top10_longest_by_e5"]:
        print(f"  {row['rank']:>2}. {row['id']:<22} E5={row['e5_tokens_with_prefix']:>5}  BGE-M3={row['bge_m3_tokens']:>5}  {row['heading']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path")
    args = parser.parse_args()

    records = load_records()
    print(f"Loaded {len(records)} records from {PROCESSED_DIR}")
    result = analyse(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print_summary(result)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
