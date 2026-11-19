"""Check data/processed/*.jsonl after chunking. Run from the repo root:
    python src/validate_records.py
Exit code 0 = every check passed, 1 = at least one check failed.
"""
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BACKUP_DIR = ROOT / "data" / "processed_v0_backup"  # local only, not in Git
BASE = "https://www.gesetze-im-internet.de"

FIELDS = {"id", "law", "law_title", "area", "section", "heading",
          "text", "source_url", "retrieval_date"}

# slug -> (clean law name, area label, expected number of records)
EXPECTED = {
    "arbzg": ("ArbZG", "employment", 26),
    "aufenthg_2004": ("AufenthG", "visa", 199),
    "beschv_2013": ("BeschV", "work", 45),
    "bgb": ("BGB", "housing", 105),
    "milog": ("MiLoG", "employment", 23),
}
BGB_RANGE = (535, 580)

SECTION_RE = re.compile(r"^§\s*(\d+)\s*([a-z]?)$")
GLUE_NUMBER = re.compile(r"[a-zäöüß]\d+\.")                      # e.g. "und2."
GLUE_LETTER = re.compile(r"(?:^|\s)[a-z]{1,2}\)[A-Za-zÄÖÜäöüß]")  # e.g. "a)während"

passed = 0
failed = 0


def check(name, bad, note=""):
    """bad = list of offending items; empty list means the check passed."""
    global passed, failed
    if bad:
        failed += 1
    else:
        passed += 1
    print(f"[{'FAIL' if bad else 'PASS'}] {name}" + (f"  ({note})" if note else ""))
    for item in bad[:5]:
        print("         ", item)
    if len(bad) > 5:
        print(f"          ... and {len(bad) - 5} more")


def skip(name, reason):
    print(f"[SKIP] {name}  ({reason})")


def label(slug, r):
    return f"{slug} {r.get('section')}"


def squash(s):
    return s.replace(" ", "")


def snippet(text, m):
    return "..." + text[max(0, m.start() - 15):m.end() + 15].replace("\n", " ") + "..."


# ---------- load ----------
by_law = {}
for slug in EXPECTED:
    path = PROCESSED_DIR / f"{slug}.jsonl"
    by_law[slug] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            by_law[slug] = [json.loads(line) for line in f if line.strip()]
records = [(slug, r) for slug, recs in by_law.items() for r in recs]
print(f"Loaded {len(records)} records from {PROCESSED_DIR}\n")

# ---------- structure ----------
check("record count per law",
      [f"{s}: expected {EXPECTED[s][2]}, found {len(recs)}"
       for s, recs in by_law.items() if len(recs) != EXPECTED[s][2]],
      f"expected total {sum(v[2] for v in EXPECTED.values())}")
check("exactly the 9 expected fields",
      [label(s, r) for s, r in records if set(r) != FIELDS])
check("no empty values",
      [label(s, r) for s, r in records
       if any(not isinstance(r.get(f), str) or not r.get(f).strip() for f in FIELDS)])

# ---------- identifiers and metadata ----------
dupes = [i for i, n in Counter(r.get("id") for _, r in records).items() if n > 1]
check("ids are unique", dupes)

bad_ids, bad_urls, bad_meta = [], [], []
for slug, r in records:
    m = SECTION_RE.match(r.get("section", ""))
    if not m:
        bad_ids.append(f"{label(slug, r)}: unreadable section label")
        continue
    key = f"{m.group(1)}{m.group(2)}"
    if r.get("id") != f"{slug}__{key}":
        bad_ids.append(f"{label(slug, r)}: id is {r.get('id')}")
    if r.get("source_url") != f"{BASE}/{slug}/__{key}.html":
        bad_urls.append(f"{label(slug, r)}: {r.get('source_url')}")
    if (r.get("law"), r.get("area")) != EXPECTED[slug][:2]:
        bad_meta.append(f"{label(slug, r)}: law={r.get('law')}, area={r.get('area')}")
check("id matches file and section number", bad_ids)
check("source_url matches the official URL pattern", bad_urls)
check("clean law name and area label per law", bad_meta)

bad_dates = []
raw_missing = []
for slug, r in records:
    try:
        date.fromisoformat(r.get("retrieval_date", ""))
    except ValueError:
        bad_dates.append(f"{label(slug, r)}: not a date: {r.get('retrieval_date')}")
xml_dates = {}
for slug in EXPECTED:
    xml_files = sorted((RAW_DIR / slug).glob("*.xml"))
    if xml_files:
        xml_dates[slug] = date.fromtimestamp(xml_files[0].stat().st_mtime).isoformat()
    else:
        raw_missing.append(slug)
check("retrieval_date is a valid date", bad_dates)
if raw_missing:
    skip("retrieval_date equals the XML file's date", f"no raw XML for: {', '.join(raw_missing)}")
bad = [f"{label(s, r)}: {r.get('retrieval_date')} vs XML {xml_dates[s]}"
       for s, r in records if s in xml_dates and r.get("retrieval_date") != xml_dates[s]]
check("retrieval_date equals the XML file's date", bad)

# ---------- text quality ----------
bad_glue, bad_space, bad_repealed, bad_bgb = [], [], [], []
for slug, r in records:
    text, heading = r.get("text", ""), r.get("heading", "")
    for rx in (GLUE_NUMBER, GLUE_LETTER):
        m = rx.search(text) or rx.search(heading)
        if m:
            bad_glue.append(f"{label(slug, r)}: {snippet(text, m) if rx.search(text) else heading}")
            break
    if "  " in text or "  " in heading or "\r" in text:
        bad_space.append(label(slug, r))
    if " ".join(text.split()) == "(weggefallen)":
        bad_repealed.append(label(slug, r))
    if slug == "bgb":
        m = SECTION_RE.match(r.get("section", ""))
        if m and not (BGB_RANGE[0] <= int(m.group(1)) <= BGB_RANGE[1]):
            bad_bgb.append(label(slug, r))
check("no glued list markers (e.g. 'und2.', 'a)während')", bad_glue)
check("no double spaces or carriage returns", bad_space)
check("no repealed-only ('(weggefallen)') records", bad_repealed)
check("BGB sections are inside 535-580", bad_bgb)

# ---------- legal text unchanged versus the original backup ----------
if BACKUP_DIR.exists():
    old = {}
    for path in sorted(BACKUP_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                old[(path.stem, r["section"])] = r
    new = {(s, r.get("section")): r for s, r in records}
    check("only repealed records were dropped vs backup",
          [f"{k}: {old[k]['text'][:40]!r}" for k in old.keys() - new.keys()
           if " ".join(old[k]["text"].split()) != "(weggefallen)"],
          f"{len(old.keys() - new.keys())} dropped")
    check("no records added vs backup", [str(k) for k in new.keys() - old.keys()])
    changed = []
    for k in old.keys() & new.keys():
        o, n = old[k], new[k]
        if squash(o["text"]) != squash(n["text"]):
            changed.append(f"{k}: text changed beyond spaces")
        if squash(o["heading"]) != squash(n["heading"]):
            changed.append(f"{k}: heading changed beyond spaces")
        for f in ("law_title", "section", "source_url", "retrieval_date"):
            if o[f] != n[f]:
                changed.append(f"{k}: {f} changed")
    check("legal text and other fields unchanged vs backup (apart from spaces)", changed)
else:
    skip("comparison with original backup", f"{BACKUP_DIR.name} not found")

# ---------- information ----------
print("\nRecords per area:", dict(Counter(r.get("area") for _, r in records)))
long_ones = sum(1 for _, r in records if len(r.get("text", "")) > 1800)
print(f"Records longer than 1800 characters: {long_ones} (information only)")
print("\nOpen these 3 URLs in your browser and compare the text:")
wanted = ["aufenthg_2004__16b", "bgb__551", "milog__1"]
for _, r in records:
    if r.get("id") in wanted:
        print("  ", r["id"], "->", r["source_url"])

print(f"\n{passed} checks passed, {failed} failed")
sys.exit(1 if failed else 0)