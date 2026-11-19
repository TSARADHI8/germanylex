"""Split long legal sections into smaller chunks -> data/processed_chunks/.

Input : data/processed/*.jsonl  (one record per section; read only, never modified)
Output: data/processed_chunks/<law>.jsonl  (one record per chunk) + chunking_manifest.json

How it works
  1. A section that already fits into --max-tokens stays ONE chunk.
  2. Otherwise it is split at paragraph (Absatz) boundaries and consecutive paragraphs are
     packed together until a chunk would exceed --target-tokens.
  3. A paragraph that is itself longer than --max-tokens is split at the next finer boundary:
     numbered list items ("1.", "2a."), lettered items ("a)"), sentences, ; or :, commas and
     finally single spaces. Text is never cut off.
  4. Very small chunks (< --min-tokens) are merged into a neighbour when the result still fits.
  5. Chunks do not overlap. Every chunk stores char_start/char_end into the original section
     text and chunk["text"] == original_text[char_start:char_end], so nothing can be lost,
     duplicated or changed without the validation noticing.

Tokens are counted with the real tokenizer (same XLM-RoBERTa vocabulary for E5 and BGE-M3).
Use --counter chars only for a rough estimate without downloading the tokenizer.

Run from the repo root:
    python src/chunk_documents.py                 # chunk + validate + write
    python src/chunk_documents.py --validate-only # re-check files already written
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "processed_chunks"
MANIFEST = "chunking_manifest.json"

LAWS = ["arbzg", "aufenthg_2004", "beschv_2013", "bgb", "milog"]
KEEP_FIELDS = ["law", "law_title", "area", "section", "heading", "source_url", "retrieval_date"]
CHUNK_FIELDS = (["chunk_id", "section_id"] + KEEP_FIELDS +
                ["chunk_index", "n_chunks", "paragraphs", "split_method", "char_start", "char_end",
                 "token_count", "text"])

E5_LIMIT = 512
E5_OVERHEAD = 4          # "passage: " (2 tokens) + <s> and </s>
CHARS_PER_TOKEN = 4.5    # only used by --counter chars (measured: 4.3-5.1 on our records)

LEVELS = ["paragraph", "list_item", "letter_item", "sentence", "clause", "comma", "word"]
ABBREVIATIONS = {"nr", "nrn", "abs", "art", "anl", "buchst", "vgl", "bzw", "ggf", "ca", "usw",
                 "etc", "dr", "prof", "z", "b", "d", "h", "u", "a", "s", "v", "m", "e", "g"}
LIST_LEAD = re.compile(r"(?:[,;:.]|\bund|\boder)$")


# --------------------------------------------------------------------------- tokens
class TokenCounter:
    def __init__(self, mode: str, model_id: str):
        self.mode, self.model_id, self.cache = mode, model_id, {}
        if mode == "tokenizer":
            try:
                from transformers import AutoTokenizer
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error()
                self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
                self.tok.model_max_length = 10**9      # we count only, never truncate
            except Exception as exc:  # noqa: BLE001
                sys.exit(f"Could not load the tokenizer '{model_id}': {exc}\n"
                         "Install it with 'pip install transformers sentencepiece' or use --counter chars "
                         "(rough estimate).")

    def __call__(self, text: str) -> int:
        n = self.cache.get(text)
        if n is None:
            if self.mode == "tokenizer":
                n = len(self.tok(text, add_special_tokens=False)["input_ids"])
            else:
                n = math.ceil(len(text) / CHARS_PER_TOKEN)
            self.cache[text] = n
        return n


# --------------------------------------------------------------------------- splitting
def find_boundaries(seg: str, level: str) -> list[tuple[int, int]]:
    """Whitespace runs (start, end) inside seg where it may be cut at this level."""
    out = []
    if level == "paragraph":
        return [(m.start(), m.end()) for m in re.finditer(r"\s*\n\s*", seg)]
    if level == "list_item":
        for m in re.finditer(r"\s+(?=\d{1,2}[a-z]?\.\s)", seg):
            if LIST_LEAD.search(seg[:m.start()]):
                out.append((m.start(), m.end()))
    elif level == "letter_item":
        for m in re.finditer(r"\s+(?=([a-z]{1,2})\)\s)", seg):
            prev = seg[:m.start()]
            if LIST_LEAD.search(prev) or (m.group(1) == "a" and not prev.endswith("Buchstabe")):
                out.append((m.start(), m.end()))
    elif level == "sentence":
        for m in re.finditer(r"(?<=\.)\s+(?=[A-ZÄÖÜ„\"(])", seg):
            token = re.search(r"(\S+)\.$", seg[:m.start()])
            word = token.group(1).lower().strip("(„\"") if token else ""
            if word in ABBREVIATIONS or re.fullmatch(r"\d+[a-z]?", word):
                continue
            out.append((m.start(), m.end()))
    elif level == "clause":
        out = [(m.start(), m.end()) for m in re.finditer(r"(?<=[;:])\s+", seg)]
    elif level == "comma":
        out = [(m.start(), m.end()) for m in re.finditer(r"(?<=,)\s+", seg)]
    else:  # word
        out = [(m.start(), m.end()) for m in re.finditer(r"\s+", seg)]
    return out


class Chunker:
    def __init__(self, count, target: int, max_tokens: int, min_tokens: int):
        self.count, self.target, self.max, self.min = count, target, max_tokens, min_tokens
        self.text = ""

    def tokens(self, s: int, e: int) -> int:
        return self.count(self.text[s:e])

    def pieces(self, a: int, b: int, level_idx: int) -> list[tuple[int, int]]:
        seg = self.text[a:b]
        cuts = find_boundaries(seg, LEVELS[level_idx])
        out, prev = [], 0
        for ws, we in cuts:
            if ws > prev:
                out.append((a + prev, a + ws))
            prev = we
        if prev < len(seg):
            out.append((a + prev, b))
        return out

    def split_span(self, a: int, b: int, level_idx: int) -> list[list]:
        """Chunk a span that is too long, starting at the given boundary level."""
        while level_idx < len(LEVELS):
            parts = self.pieces(a, b, level_idx)
            if len(parts) > 1:
                return self.pack(parts, level_idx)
            level_idx += 1
        return [[a, b, "word"]]       # a single word longer than the limit: keep it whole

    def pack(self, parts, level_idx) -> list[list]:
        out, cur = [], None
        name = LEVELS[level_idx]
        for s, e in parts:
            if self.tokens(s, e) > self.max:          # too long on its own: go one level finer
                if cur:
                    out.append(cur)
                    cur = None
                out.extend(self.split_span(s, e, level_idx + 1))
            elif cur is None:
                cur = [s, e, name]
            elif self.tokens(cur[0], e) <= self.target:
                cur[1] = e
            else:
                out.append(cur)
                cur = [s, e, name]
        if cur:
            out.append(cur)
        return out

    def merge_small(self, chunks: list[list]) -> list[list]:
        changed = True
        while changed and len(chunks) > 1:
            changed = False
            for i, c in enumerate(chunks):
                if self.tokens(c[0], c[1]) >= self.min:
                    continue
                for j in (i - 1, i + 1):                   # try previous, then next neighbour
                    if 0 <= j < len(chunks):
                        lo, hi = chunks[min(i, j)], chunks[max(i, j)]
                        if self.tokens(lo[0], hi[1]) <= self.max:
                            deeper = max(c[2], chunks[j][2], key=LEVELS.index)
                            merged = [lo[0], hi[1], deeper]
                            chunks[min(i, j):max(i, j) + 1] = [merged]
                            changed = True
                            break
                if changed:
                    break
        return chunks

    def chunk_section(self, text: str) -> list[list]:
        self.text = text
        if self.tokens(0, len(text)) <= self.max:
            return [[0, len(text), "whole_section"]]
        parts = self.pieces(0, len(text), 0)
        chunks = self.pack(parts, 0) if len(parts) > 1 else self.split_span(0, len(text), 1)
        return self.merge_small(chunks)


def paragraph_labels(text: str, s: int, e: int) -> list[str]:
    """Absatz numbers like '(2)' of the paragraphs this chunk touches (empty if unnumbered)."""
    labels, pos = [], 0
    for line in text.split("\n"):
        end = pos + len(line)
        if pos < e and end > s:
            m = re.match(r"\((\d+[a-z]?)\)", line)
            if m:
                labels.append(f"({m.group(1)})")
        pos = end + 1
    return labels


def build_chunks(records: list[dict], chunker: Chunker) -> list[dict]:
    out = []
    for rec in records:
        text = rec["text"]
        spans = chunker.chunk_section(text)
        for i, (s, e, how) in enumerate(spans, 1):
            piece = text[s:e]
            out.append({
                "chunk_id": f"{rec['id']}__c{i:02d}", "section_id": rec["id"],
                **{k: rec[k] for k in KEEP_FIELDS},
                "chunk_index": i, "n_chunks": len(spans),
                "paragraphs": paragraph_labels(text, s, e), "split_method": how,
                "char_start": s, "char_end": e, "token_count": chunker.count(piece), "text": piece,
            })
    return out


# --------------------------------------------------------------------------- IO and validation
def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(s: str) -> str:
    return " ".join(s.split())


class Report:
    def __init__(self):
        self.passed = self.failed = 0

    def check(self, name, bad, note=""):
        ok = not bad
        self.passed += ok
        self.failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({note})" if note else ""))
        for item in bad[:5]:
            print("         ", item)
        if len(bad) > 5:
            print(f"          ... and {len(bad) - 5} more")


def validate(originals: dict[str, list[dict]], chunks: dict[str, list[dict]], count, args) -> bool:
    rep = Report()
    orig = {r["id"]: r for recs in originals.values() for r in recs}
    allc = [c for cs in chunks.values() for c in cs]
    by_sec: dict[str, list[dict]] = {}
    for c in allc:
        by_sec.setdefault(c["section_id"], []).append(c)

    rep.check("every original section has at least one chunk",
              sorted(set(orig) - set(by_sec)), f"{len(orig)} sections")
    rep.check("no chunk refers to an unknown section", sorted(set(by_sec) - set(orig)))
    rep.check("chunk files hold exactly the expected fields",
              [c["chunk_id"] for c in allc if set(c) != set(CHUNK_FIELDS)])
    rep.check("chunk ids are unique",
              [i for i, n in Counter(c["chunk_id"] for c in allc).items() if n > 1], f"{len(allc)} chunks")
    rep.check("no chunk has empty text", [c["chunk_id"] for c in allc if not c["text"].strip()])
    rep.check("metadata equals the original record",
              [c["chunk_id"] for c in allc
               if any(c[k] != orig[c["section_id"]][k] for k in KEEP_FIELDS)
               or c["section_id"] not in orig])
    rep.check("chunk files sit in the same law file as their section",
              [c["chunk_id"] for slug, cs in chunks.items() for c in cs
               if c["section_id"].split("__")[0] != slug])

    bad_order, bad_text, bad_gap, bad_norm, bad_count = [], [], [], [], []
    for sid, cs in by_sec.items():
        if sid not in orig:
            continue
        text = orig[sid]["text"]
        cs = sorted(cs, key=lambda c: c["chunk_index"])
        if [c["chunk_index"] for c in cs] != list(range(1, len(cs) + 1)) or any(c["n_chunks"] != len(cs) for c in cs):
            bad_order.append(sid)
        if any(c["chunk_id"] != f"{sid}__c{c['chunk_index']:02d}" for c in cs):
            bad_order.append(sid + " (chunk_id)")
        if any(text[c["char_start"]:c["char_end"]] != c["text"] for c in cs):
            bad_text.append(sid)
        gaps = [text[:cs[0]["char_start"]]] + [text[a["char_end"]:b["char_start"]] for a, b in zip(cs, cs[1:])] \
            + [text[cs[-1]["char_end"]:]]
        if any(g.strip() for g in gaps) or any(a["char_end"] > b["char_start"] for a, b in zip(cs, cs[1:])):
            bad_gap.append(sid)
        if norm(" ".join(c["text"] for c in cs)) != norm(text):
            bad_norm.append(sid)
        if any(c["token_count"] != count(c["text"]) for c in cs):
            bad_count.append(sid)
    rep.check("chunk numbering (1..n) and chunk ids are consistent", bad_order)
    rep.check("chunk text equals original_text[char_start:char_end]", bad_text)
    rep.check("chunks are ordered, do not overlap, and skip only whitespace", bad_gap)
    rep.check("joined chunk texts equal the original text (whitespace-normalised)", bad_norm)
    rep.check("stored token_count matches a fresh count", bad_count)
    rep.check("no chunk exceeds the hard token limit",
              [f"{c['chunk_id']}: {c['token_count']}" for c in allc if c["token_count"] > args.max_tokens],
              f"max {args.max_tokens}")
    rep.check("sections that fit stay as exactly one chunk",
              [sid for sid, r in orig.items()
               if count(r["text"]) <= args.max_tokens and len(by_sec.get(sid, [])) != 1])
    rep.check("sections that do not fit are really split",
              [sid for sid, r in orig.items()
               if count(r["text"]) > args.max_tokens and len(by_sec.get(sid, [])) < 2])
    rep.check(f"heading + chunk fits the E5 limit of {E5_LIMIT} tokens (with prefix and special tokens)",
              [f"{c['chunk_id']}: {count(c['heading'] + chr(10) + c['text']) + E5_OVERHEAD}" for c in allc
               if count(c["heading"] + "\n" + c["text"]) + E5_OVERHEAD > E5_LIMIT])
    print(f"\n{rep.passed} checks passed, {rep.failed} failed")
    return rep.failed == 0


def summary(originals, chunks, count):
    allc = [c for cs in chunks.values() for c in cs]
    secs = [r for recs in originals.values() for r in recs]
    tok = [c["token_count"] for c in allc]
    split_secs = sum(1 for c in allc if c["n_chunks"] > 1 and c["chunk_index"] == 1)
    print("\n=== Chunking summary ===")
    print(f"sections: {len(secs)} | chunks: {len(allc)} | sections split: {split_secs} "
          f"| sections kept whole: {len(secs) - split_secs}")
    print(f"chunk tokens: min {min(tok)} | median {int(statistics.median(tok))} | "
          f"mean {statistics.fmean(tok):.0f} | max {max(tok)}")
    print("chunks per law:", {k: len(v) for k, v in chunks.items()})
    print("split methods:", dict(Counter(c["split_method"] for c in allc)))
    big = sorted({c["section_id"]: c["n_chunks"] for c in allc if c["n_chunks"] > 1}.items(),
                 key=lambda kv: -kv[1])[:5]
    print("most chunks per section:", big)
    return {"sections": len(secs), "chunks": len(allc), "sections_split": split_secs,
            "chunk_tokens": {"min": min(tok), "median": statistics.median(tok),
                             "mean": round(statistics.fmean(tok), 1), "max": max(tok)},
            "chunks_per_law": {k: len(v) for k, v in chunks.items()},
            "split_methods": dict(Counter(c["split_method"] for c in allc))}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-tokens", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=450, help="hard limit for chunk text")
    ap.add_argument("--min-tokens", type=int, default=100, help="smaller chunks are merged if they still fit")
    ap.add_argument("--counter", choices=["tokenizer", "chars"], default="tokenizer")
    ap.add_argument("--tokenizer", default="intfloat/multilingual-e5-base")
    ap.add_argument("--input-dir", type=Path, default=IN_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    if not (0 < args.min_tokens < args.target_tokens <= args.max_tokens):
        sys.exit("Need 0 < min-tokens < target-tokens <= max-tokens.")

    files = {slug: args.input_dir / f"{slug}.jsonl" for slug in LAWS}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        sys.exit("Missing input files:\n  " + "\n  ".join(missing))
    hashes_before = {s: sha256(p) for s, p in files.items()}
    originals = {s: read_jsonl(p) for s, p in files.items()}
    print(f"Read {sum(map(len, originals.values()))} sections from {args.input_dir}")
    count = TokenCounter(args.counter, args.tokenizer)

    if args.validate_only:
        miss = [s for s in LAWS if not (args.output_dir / f"{s}.jsonl").exists()]
        if miss:
            sys.exit(f"No chunk files for {miss} in {args.output_dir}; run without --validate-only first.")
        chunks = {s: read_jsonl(args.output_dir / f"{s}.jsonl") for s in LAWS}
    else:
        chunker = Chunker(count, args.target_tokens, args.max_tokens, args.min_tokens)
        chunks = {s: build_chunks(originals[s], chunker) for s in LAWS}

    print("\n=== Validation (in memory) ===" if not args.validate_only else "\n=== Validation (files on disk) ===")
    ok = validate(originals, chunks, count, args)
    stats = summary(originals, chunks, count)
    if not ok:
        print("\nValidation failed; nothing was written." if not args.validate_only else "\nValidation failed.")
        return 1

    if not args.validate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for slug, cs in chunks.items():
            with open(args.output_dir / f"{slug}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
                for c in cs:
                    fh.write(json.dumps({k: c[k] for k in CHUNK_FIELDS}, ensure_ascii=False) + "\n")
        manifest = {"settings": {"target_tokens": args.target_tokens, "max_tokens": args.max_tokens,
                                 "min_tokens": args.min_tokens, "counter": args.counter,
                                 "tokenizer": args.tokenizer if args.counter == "tokenizer" else None,
                                 "tokens_are_estimates": args.counter == "chars"},
                    "stats": stats, "input_sha256": hashes_before}
        with open(args.output_dir / MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        print(f"\nWrote {sum(map(len, chunks.values()))} chunks to {args.output_dir}")
        chunks = {s: read_jsonl(args.output_dir / f"{s}.jsonl") for s in LAWS}   # round-trip check
        print("\n=== Validation (files re-read from disk) ===")
        if not validate(originals, chunks, count, args):
            return 1

    unchanged = {s: sha256(p) for s, p in files.items()} == hashes_before
    print(f"\nOriginal data/processed files unchanged: {unchanged}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
