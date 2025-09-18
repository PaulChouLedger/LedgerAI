from pdfminer.high_level import extract_text

def parse_pdf_file(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"[PDFParser] ❌ Error parsing {pdf_path}: {e}")
        return ""
