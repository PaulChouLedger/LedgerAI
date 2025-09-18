import os
from bs4 import BeautifulSoup
from .pdf_parser import parse_pdf_file

# === Directories ===
INPUT_DIR = "data/input"
PARSED_DIR = "data/parsed"

os.makedirs(PARSED_DIR, exist_ok=True)

def parse_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def parse_html_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n")

def save_parsed_text(file_path, text):
    filename = os.path.basename(file_path)
    filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(PARSED_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Ingestion] ✅ Parsed and saved: {filename}")

def ingest_file(file_path):
    filename = os.path.basename(file_path)
    parsed_path = os.path.join(PARSED_DIR, os.path.splitext(filename)[0] + ".txt")

    if os.path.exists(parsed_path):
        print(f"[Ingestion] ⏩ Skipping {file_path} — already parsed.")
        return

    if file_path.endswith(".pdf"):
        text = parse_pdf_file(file_path)
    elif file_path.endswith(".txt"):
        text = parse_txt_file(file_path)
    elif file_path.endswith((".html", ".htm")):
        text = parse_html_file(file_path)
    else:
        print(f"[Ingestion] ❌ Unsupported file format: {file_path}")
        return

    save_parsed_text(file_path, text)

def ingest_all_supported_files():
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith((".txt", ".pdf", ".html", ".htm")):
            file_path = os.path.join(INPUT_DIR, filename)
            ingest_file(file_path)
