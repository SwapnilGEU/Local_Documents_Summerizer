import pymupdf
import pymupdf4llm
from tqdm import tqdm

PDF_PATH = "data/raw/MachineLearningTomMitchell.pdf"
OUTPUT_PATH = "data/processed/ml_book.md"


print("Opening PDF...")

doc = pymupdf.open(PDF_PATH)

print(f"Total pages: {len(doc)}")
print("Starting extraction...\n")


all_pages = []

for page_num in tqdm(
    range(len(doc)),
    desc="Extracting PDF",
    unit="page"
):
    # Extract one page using PyMuPDF4LLM
    page_data = pymupdf4llm.to_markdown(
        doc,
        pages=[page_num],
        page_chunks=True
    )

    all_pages.extend(page_data)


doc.close()


print("\nSaving Markdown...")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for page in all_pages:

        page_number = page["metadata"]["page"] + 1

        f.write(
            f"\n\n<!-- PAGE {page_number} -->\n\n"
        )

        f.write(page["text"])


print("\nExtraction completed!")
print(f"Saved to: {OUTPUT_PATH}")