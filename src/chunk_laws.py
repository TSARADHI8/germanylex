"""Split downloaded law XML into one chunk per section (§) with metadata -> data/processed/."""
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
BASE = "https://www.gesetze-im-internet.de"

# Only the tenancy part of the BGB (§§ 535-580a) is used in the first version.
BGB_RANGE = (535, 580)


def text_of(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def section_number(enbez: str):
    m = re.match(r"§\s*(\d+)\s*([a-z]?)", enbez or "")
    return (int(m.group(1)), m.group(2)) if m else None


def section_url(slug: str, enbez: str) -> str:
    m = section_number(enbez)
    return f"{BASE}/{slug}/__{m[0]}{m[1]}.html" if m else f"{BASE}/{slug}/"


def chunk_law(slug: str):
    xml_files = sorted((RAW_DIR / slug).glob("*.xml"))
    if not xml_files:
        print(f"[SKIP] {slug}: no XML in data/raw/{slug}")
        return []
    root = ET.parse(xml_files[0]).getroot()
    today = date.today().isoformat()
    chunks = []
    law_abbr = law_title = None
    for norm in root.iter("norm"):
        meta = norm.find("metadaten")
        if meta is None:
            continue
        abbr = text_of(meta.find("jurabk"))
        if law_abbr is None:
            law_abbr = abbr
            law_title = text_of(meta.find("langue"))
        enbez = text_of(meta.find("enbez"))
        num = section_number(enbez)
        if num is None:  # skip headings, annexes, table of contents
            continue
        if slug == "bgb" and not (BGB_RANGE[0] <= num[0] <= BGB_RANGE[1]):
            continue
        content = norm.find("./textdaten/text/Content")
        paras = [text_of(p) for p in content.iter("P")] if content is not None else []
        body = "\n".join(p for p in paras if p) or text_of(content)
        if not body:
            continue
        chunks.append({
            "law": abbr or law_abbr,
            "law_title": law_title,
            "section": enbez,
            "heading": text_of(meta.find("titel")),
            "text": body,
            "source_url": section_url(slug, enbez),
            "retrieval_date": today,
        })
    return chunks


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slugs = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir()) if RAW_DIR.exists() else []
    if not slugs:
        print("No raw data. Run src/download_laws.py first.")
        return 1
    total = 0
    for slug in slugs:
        chunks = chunk_law(slug)
        with open(OUT_DIR / f"{slug}.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[OK]   {slug}: {len(chunks)} sections")
        total += len(chunks)
    print(f"Total: {total} sections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
