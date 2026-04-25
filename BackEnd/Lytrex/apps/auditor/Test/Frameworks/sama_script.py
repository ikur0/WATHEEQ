"""
SAMA PDF Extractor
------------------
Extracts cybersecurity controls from SAMA_EN_5888_VER1.pdf and saves them to SAMA.json.

Layout (confirmed from document):
  - Sections are plain text, NOT tables
  - Section headers look like:  3.1.1  Cyber Security Governance
  - Under each section there are blocks:
        Principle            -> SKIP
        Objective            -> SKIP
        Control considerations -> CAPTURE everything below until next section
  - Only store 3-part (3.1.1) and 4-part (3.1.1.1) IDs
  - Skip 2-part (3.1) and 1-part IDs

Usage:
    pip install pdfplumber
    python extract_sama.py                                 # default path
    python extract_sama.py path/to/SAMA_EN_5888_VER1.pdf  # custom path

Output: SAMA.json
"""

import json
import re
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    sys.exit("pdfplumber is required.  Run:  pip install pdfplumber")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_PDF = "SAMA/SAMA_EN_5888_VER1.pdf"
OUTPUT_FILE = "SAMA.json"

# 3-part: 3.1.1   4-part: 3.1.1.1  (dot or dash separated)
SECTION_RE = re.compile(
    r"^(\d+\.\d+\.\d+(?:\.\d+)?|\d+-\d+-\d+(?:-\d+)?)\s+(.+)$"
)

# Block labels to skip content under
SKIP_BLOCK_RE = re.compile(
    r"^(principle|objective|purpose)\s*:?\s*$", re.IGNORECASE
)

# Block label that introduces capturable content
CAPTURE_BLOCK_RE = re.compile(
    r"^control\s+considerations\s*:?\s*$", re.IGNORECASE
)

# Page footer / header noise to discard
FOOTER_RE = re.compile(
    r"^(version\s+\d|page\s+\d+|\d+\s+of\s+\d+|www\.|http)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_3or4_part(section_id: str) -> bool:
    parts = re.split(r"[.\-]", section_id)
    return len(parts) in (3, 4)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_sections(pdf_path: str) -> list:
    sections = []
    current = None

    # State machine:
    #   'idle'    -> between known blocks, ignore lines
    #   'skip'    -> inside Principle / Objective block, ignore lines
    #   'capture' -> inside Control considerations, keep lines
    state = "idle"

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"  Pages: {total_pages}")

        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(layout=False) or ""
            lines = raw_text.splitlines()

            for line in lines:
                line = clean_line(line)
                if not line:
                    continue

                # Drop footer / header noise
                if FOOTER_RE.match(line):
                    continue

                # ── New section header? ──────────────────────────────────────
                m = SECTION_RE.match(line)
                if m:
                    sec_id = m.group(1)
                    title  = clean_line(m.group(2))

                    if is_3or4_part(sec_id):
                        # Save previous
                        if current is not None:
                            current["page_range"][1] = page_num
                            sections.append(current)

                        current = {
                            "section_id": "",
                            "section":    sec_id,
                            "title":      title,
                            "page_range": [page_num, page_num],
                            "text":       "",
                            "depends_on": [],
                        }
                        state = "idle"
                    else:
                        # 2-part or 1-part heading — reset state, no new section
                        state = "idle"
                    continue

                # ── Block label detection ────────────────────────────────────
                if SKIP_BLOCK_RE.match(line):
                    state = "skip"
                    continue

                if CAPTURE_BLOCK_RE.match(line):
                    state = "capture"
                    continue

                # ── Content lines ────────────────────────────────────────────
                if current is None:
                    continue

                if state == "capture":
                    if current["text"]:
                        current["text"] += "\n" + line
                    else:
                        current["text"] = line
                # state == "skip" or "idle" -> discard line

        # Flush last open section
        if current is not None:
            current["page_range"][1] = total_pages
            sections.append(current)

    # Auto-increment IDs
    for idx, sec in enumerate(sections):
        sec["section_id"] = f"SEC_{idx:04d}"

    return sections


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF

    if not Path(pdf_path).exists():
        sys.exit(
            f"PDF not found: {pdf_path}\n"
            f"Usage: python extract_sama.py [path/to/SAMA_EN_5888_VER1.pdf]"
        )

    print(f"Reading: {pdf_path}")
    sections = extract_sections(pdf_path)
    print(f"Extracted {len(sections)} control sections.")

    output = {
        "framework":   "SAMA",
        "source_file": Path(pdf_path).name,
        "sections":    sections,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved -> {OUTPUT_FILE}")

    # Preview first 3
    print("\n--- Preview (first 3 sections) ---")
    for sec in sections[:3]:
        print(json.dumps(sec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()