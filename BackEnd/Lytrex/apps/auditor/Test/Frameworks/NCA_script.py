import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise SystemExit(
        "PyMuPDF is required. Install it with: pip install pymupdf"
    ) from exc


# ----------------------------
# Config
# ----------------------------
PDF_PATH = "NCA/ncs_en.pdf"          # change if needed
OUTPUT_JSON = "NCA.json"
MIN_SECTION_CHARS = 120          # merge tiny false-positive headings into previous section
MERGE_SHORT_SECTIONS = True


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Section:
    section_id: str
    title: str
    level: int
    start_page: int
    end_page: int
    text: str
    parent_id: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------
def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if re.fullmatch(r"\d+", s):
        return True
    if re.fullmatch(r"Page\s+\d+(\s+of\s+\d+)?", s, flags=re.I):
        return True
    return False


def clean_page_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if not is_noise_line(ln)]
    return normalize_text("\n".join(lines))


def extract_pages(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(clean_page_text(page.get_text("text")))
    doc.close()
    return pages


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if len(s) < 3 or len(s) > 180:
        return False

    # Strong numbered heading patterns: 1 / 1.2 / 1.2.3 / A.1 / 3-2
    numbered = re.match(r"^(?:[A-Z]\.)?\d+(?:[.\-]\d+)*[.)]?\s+.+$", s)

    # All-caps heading-like lines
    upperish = (
        len(s.split()) <= 12
        and any(c.isalpha() for c in s)
        and s == s.upper()
        and not re.fullmatch(r"[A-Z]{1,5}", s)
    )

    # Title-case / heading keywords
    keywordish = re.match(
        r"^(?:Domain|Section|Subsection|Control|Requirement|Clause|Article|Chapter)\b",
        s,
        flags=re.I,
    )

    # Avoid obvious body lines ending with punctuation
    ends_like_sentence = s.endswith((".", ":", ";", ","))

    return bool((numbered or upperish or keywordish) and not ends_like_sentence)


def heading_level(line: str) -> int:
    s = line.strip()
    m = re.match(r"^(?:[A-Z]\.)?(\d+(?:[.\-]\d+)*)", s)
    if m:
        parts = re.split(r"[.\-]", m.group(1))
        return max(1, len(parts))
    if s == s.upper():
        return 1
    return 1


def extract_heading_title(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^(?:[A-Z]\.)?\d+(?:[.\-]\d+)*[.)]?\s*", "", s)
    return s.strip()


def build_section_id(index: int) -> str:
    return f"SEC_{index:04d}"


def find_parent_id(sections: List[Section], current_index: int) -> Optional[str]:
    current = sections[current_index]
    for j in range(current_index - 1, -1, -1):
        prev = sections[j]
        if prev.level < current.level:
            return prev.section_id
    return None


# ----------------------------
# Main parsing logic
# ----------------------------
def parse_sections_from_pages(pages: List[str]) -> List[Section]:
    sections: List[Section] = []
    current_title = "Document Start"
    current_level = 1
    current_start_page = 1
    buffer: List[str] = []
    section_counter = 1

    def flush(end_page: int) -> None:
        nonlocal section_counter, buffer, current_title, current_level, current_start_page
        text = normalize_text("\n".join(buffer))
        if not text:
            return
        sections.append(
            Section(
                section_id=build_section_id(section_counter),
                title=current_title,
                level=current_level,
                start_page=current_start_page,
                end_page=end_page,
                text=text,
            )
        )
        section_counter += 1
        buffer = []

    for page_no, page_text in enumerate(pages, start=1):
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        for line in lines:
            if looks_like_heading(line):
                if buffer:
                    flush(page_no)
                current_title = extract_heading_title(line) or line.strip()
                current_level = heading_level(line)
                current_start_page = page_no
                buffer = [line]
            else:
                buffer.append(line)

    if buffer:
        flush(len(pages))

    # assign parent ids
    for i in range(len(sections)):
        sections[i].parent_id = find_parent_id(sections, i)

    if MERGE_SHORT_SECTIONS:
        sections = merge_tiny_sections(sections)

    return sections


def merge_tiny_sections(sections: List[Section]) -> List[Section]:
    if not sections:
        return sections

    merged: List[Section] = [sections[0]]
    for sec in sections[1:]:
        if len(sec.text) < MIN_SECTION_CHARS:
            prev = merged[-1]
            prev.text = normalize_text(prev.text + "\n\n" + sec.title + "\n" + sec.text)
            prev.end_page = max(prev.end_page, sec.end_page)
        else:
            merged.append(sec)

    # re-number and rebuild hierarchy conservatively
    for i, sec in enumerate(merged, start=1):
        sec.section_id = build_section_id(i)
        sec.parent_id = None
    for i in range(len(merged)):
        merged[i].parent_id = find_parent_id(merged, i)
    return merged


# ----------------------------
# Export
# ----------------------------
def to_json_dict(pdf_path: str, sections: List[Section]) -> Dict[str, Any]:
    return {
        "framework": "NCA",
        "source_file": Path(pdf_path).name,
        "total_sections": len(sections),
        "sections": [asdict(s) for s in sections],
    }


def main() -> None:
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path.resolve()}")

    pages = extract_pages(str(pdf_path))
    sections = parse_sections_from_pages(pages)
    data = to_json_dict(str(pdf_path), sections)

    out_path = Path(OUTPUT_JSON)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Wrote {out_path.resolve()}")
    print(f"Total sections: {len(sections)}")
    if sections:
        print("First 5 titles:")
        for sec in sections[:5]:
            print(f"- {sec.section_id} | level={sec.level} | {sec.title}")


if __name__ == "__main__":
    main()
