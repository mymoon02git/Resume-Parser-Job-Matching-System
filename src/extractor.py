import os
import re
import pdfplumber
import docx


def extract_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_from_docx(path: str) -> str:
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def clean_text(text: str) -> str:
    text = text.replace("\u2022", " ")          # bullet symbols
    text = re.sub(r"[ \t]+", " ", text)         # extra spaces
    text = re.sub(r"\n\s*\n+", "\n", text)      # blank lines
    return text.strip()


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        raw = extract_from_pdf(path)
    elif ext == ".docx":
        raw = extract_from_docx(path)
    elif ext == ".txt":
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return clean_text(raw)


if __name__ == "__main__":
    print(extract_text("data/sample_resumes/resume1.pdf")[:1000])