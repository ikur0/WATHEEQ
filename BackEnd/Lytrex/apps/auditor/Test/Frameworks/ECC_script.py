"""
ECC PDF Extractor
-----------------
Extracts cybersecurity controls from ECC--2024-EN.pdf and saves them to ECC.json.

Usage:
    pip install pdfplumber
    python extract_ecc.py                          # uses default path
    python extract_ecc.py path/to/ECC--2024-EN.pdf # custom path

Output: ECC.json in the same directory as the script.
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

DEFAULT_PDF = "ECC/ECC--2024-EN.pdf"
OUTPUT_FILE = "ECC.json"

# Matches section IDs like  1-1  /  1-1-1  /  2-3-4  at the start of a cell
SECTION_RE = re.compile(r"^(\d+-\d+(?:-\d+)?)$")

# Top-level sections like "1-1", "1-2" (exactly two numeric parts)
TOP_SECTION_RE = re.compile(r"^\d+-\d+$")

# Rows whose first cell matches these are skipped (not saved as controls)
SKIP_LABELS = {"objective", "controls"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    """Normalize whitespace in extracted text."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_section_id(text: str) -> bool:
    return bool(SECTION_RE.match(clean(text)))


def is_top_section(section_id: str) -> bool:
    return bool(TOP_SECTION_RE.match(section_id))


def extract_rows_from_page(page) -> list[list[str]]:
    """
    Extract rows from all tables on a page.
    Each row is a list of cleaned cell strings.
    """
    rows = []
    for table in page.extract_tables():
        for row in table:
            cleaned = [clean(cell or "") for cell in row]
            # Skip fully empty rows
            if any(cleaned):
                rows.append(cleaned)
    return rows


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_sections(pdf_path: str) -> list[dict]:
    """
    Parse the PDF and return a list of section dicts.

    Each top-level section (e.g. 1-1, 1-2) becomes one entry.
    Its text is built from:
      - The title row (e.g. "1-1 | Cybersecurity Strategy")
      - All control sub-rows (e.g. 1-1-1, 1-1-2 …)
    The Objective row is intentionally skipped.
    """
    sections: list[dict] = []
    current: dict | None = None
    current_page_start: int = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            rows = extract_rows_from_page(page)

            for row in rows:
                if not row:
                    continue

                first_cell = row[0].strip()
                second_cell = row[1].strip() if len(row) > 1 else ""

                # ---- Skip header/footer/label rows -------------------------
                if first_cell.lower() in SKIP_LABELS:
                    continue

                # ---- New top-level section (e.g. "1-1") --------------------
                if is_section_id(first_cell) and is_top_section(first_cell):
                    # Save the previous section
                    if current is not None:
                        current["page_range"][1] = page_num
                        sections.append(current)

                    current = {
                        "section_id": "",          # filled later
                        "section": first_cell,
                        "title": second_cell,
                        "page_range": [page_num, page_num],
                        "text": "",
                        "depends_on": [],
                    }
                    current_page_start = page_num
                    continue

                # ---- Sub-control row (e.g. "1-1-1") ------------------------
                if is_section_id(first_cell) and not is_top_section(first_cell):
                    if current is not None:
                        # Append sub-control text
                        control_text = f"{first_cell}: {second_cell}"
                        if current["text"]:
                            current["text"] += "\n" + control_text
                        else:
                            current["text"] = control_text
                    continue

                # ---- Continuation row (no section ID in first cell) --------
                # This handles wrapped text that lands in a separate table row
                if current is not None and first_cell and not is_section_id(first_cell):
                    # Only append if it looks like body text (not a heading)
                    combined = " ".join(c for c in row if c)
                    if current["text"]:
                        current["text"] += " " + combined
                    # (if text is empty it's probably the title row body — ignore)

        # Don't forget the last open section
        if current is not None:
            current["page_range"][1] = total_pages
            sections.append(current)

    # Assign auto-increment IDs
    for idx, sec in enumerate(sections):
        sec["section_id"] = f"SEC_{idx:04d}"

    return sections


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF

    if not Path(pdf_path).exists():
        sys.exit(f"PDF not found: {pdf_path}\nUsage: python extract_ecc.py [path/to/ECC--2024-EN.pdf]")

    print(f"Reading: {pdf_path}")
    sections = extract_sections(pdf_path)
    print(f"Extracted {len(sections)} sections.")

    output = {
        "framework": "ECC",
        "source_file": Path(pdf_path).name,
        "sections": sections,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved → {OUTPUT_FILE}")

    # Quick preview
    print("\n--- Preview (first 2 sections) ---")
    for sec in sections[:2]:
        print(json.dumps(sec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()