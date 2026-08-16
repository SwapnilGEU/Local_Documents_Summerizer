
from pathlib import Path
import re
from config import MD_PATH
from config import RAW_PDF
from config import CLEAN_PATH

if MD_PATH.exists():
    print(f"Already extracted: {MD_PATH}")
else:
    import pymupdf
    import pymupdf4llm
    from tqdm import tqdm

    doc = pymupdf.open(RAW_PDF)
    print(f"Total pages: {len(doc)}")

    all_pages = []
    for page_num in tqdm(range(len(doc)), desc="Extracting PDF", unit="page"):
        page_data = pymupdf4llm.to_markdown(
            doc,
            pages=[page_num],
            page_chunks=True
        )
        all_pages.extend(page_data)

    doc.close()

    with open(MD_PATH, "w", encoding="utf-8") as f:
        for page_number, page in enumerate(all_pages, start=1):
            f.write(f"\n\n<!-- PAGE {page_number} -->\n\n")
            f.write(page["text"])

    print(f"Saved to: {MD_PATH}")

if CLEAN_PATH.exists():
    print(f"Already cleaned: {CLEAN_PATH}")
else:
    text = MD_PATH.read_text(encoding="utf-8")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n*\s*(<!-- PAGE \d+ -->)\s*\n*", r"\n\n\1\n\n", text)
    text = re.sub(r" +([,.!?;:])", r"\1", text)

    CLEAN_PATH.write_text(text.strip(), encoding="utf-8")
    print(f"Saved to: {CLEAN_PATH}")

text = CLEAN_PATH.read_text(encoding="utf-8")

HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
PAGE_RE = re.compile(r"<!-- PAGE (\d+) -->")

def strip_md_emphasis(s: str) -> str:
    return re.sub(r"[*_]+", "", s).strip()

def is_real_heading(raw_title: str) -> bool:
    title = strip_md_emphasis(raw_title)

    if re.match(r"^(FIGURE|TABLE)\b", title, re.IGNORECASE):
        return False

    if re.match(r"^\d+(\.\d+){0,4}\s+\S", title):
        return True

    letters_only = re.sub(r"[^A-Za-z]", "", title)
    if len(letters_only) >= 4 and letters_only.isupper():
        return True

    return False

structure = []
current_page = None

for line in text.splitlines():
    page_match = PAGE_RE.match(line.strip())
    if page_match:
        current_page = int(page_match.group(1))
        continue

    heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line.strip())
    if heading_match:
        level = len(heading_match.group(1))
        raw_title = heading_match.group(2)

        if is_real_heading(raw_title):
            structure.append({
                "page": current_page,
                "level": level,
                "title": strip_md_emphasis(raw_title),
            })

print(f"Real headings kept: {len(structure)}")

def find_heading_positions(text, structure):
    positions = []
    search_from = 0

    for item in structure:
        pattern = re.compile(
            r"(?m)^#{1,6}\s+\*{0,3}_{0,3}" +
            re.escape(item["title"][:40])
        )
        match = pattern.search(text, search_from)

        if match is None:
            match = pattern.search(text)

        positions.append(match.start() if match else search_from)

        if match:
            search_from = match.end()

    return positions

def heading_path(idx):
    level = structure[idx]["level"]
    path = [structure[idx]["title"]]

    for j in range(idx - 1, -1, -1):
        if structure[j]["level"] < level:
            path.insert(0, structure[j]["title"])
            level = structure[j]["level"]

        if level <= 1:
            break

    return " > ".join(path)

positions = find_heading_positions(text, structure)

sections = []

for i, item in enumerate(structure):
    start = positions[i]
    end = positions[i + 1] if i + 1 < len(positions) else len(text)

    body = text[start:end]
    body = re.sub(r"(?m)^#{1,6}\s+.+?$", "", body, count=1).strip()
    body = PAGE_RE.sub("", body).strip()

    if body:
        sections.append({
            "title": item["title"],
            "heading_path": heading_path(i),
            "page": item["page"],
            "text": body,
        })

print(f"Sections with body text: {len(sections)}")