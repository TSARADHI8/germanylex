"""Download official law texts (XML) from gesetze-im-internet.de into data/raw/."""
import io
import sys
import zipfile
from pathlib import Path

import requests

BASE = "https://www.gesetze-im-internet.de"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# slug on gesetze-im-internet.de -> short name
LAWS = {
    "aufenthg_2004": "AufenthG",
    "bgb": "BGB",
    "beschv_2013": "BeschV",
    "arbzg": "ArbZG",
    "milog": "MiLoG",
}


def download(slug: str) -> bool:
    url = f"{BASE}/{slug}/xml.zip"
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "GermanyLex-capstone/0.1"})
    except requests.RequestException as e:
        print(f"[FAIL] {slug}: {e}")
        return False
    if r.status_code != 200:
        print(f"[FAIL] {slug}: HTTP {r.status_code} for {url}")
        return False
    out = RAW_DIR / slug
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out)
    print(f"[OK]   {slug}: {len(r.content) // 1024} KB -> {out}")
    return True


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = [download(s) for s in LAWS]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
