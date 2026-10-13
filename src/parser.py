import re
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

# ---------- Load skills & build matcher ----------
def load_skills(path="data/skills.txt"):
    with open(path, encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS = load_skills()
skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
skill_matcher.add("SKILLS", [nlp.make_doc(s) for s in SKILLS])


# ---------- Individual extractors ----------
def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else None


def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[\s-]?)?(\d{10}|\d{5}[\s-]\d{5}|\d{3}[\s-]\d{3}[\s-]\d{4})", text)
    return match.group(0).strip() if match else None


def extract_name(text):
    # Name is usually within the first few lines
    header = "\n".join(text.split("\n")[:5])
    doc = nlp(header)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.title()
    # Fallback: first line
    return text.split("\n")[0].strip().title()


def extract_skills(text):
    doc = nlp(text.lower())
    matches = skill_matcher(doc)
    found = {doc[start:end].text for _, start, end in matches}
    return sorted(found)


def extract_education(text):
    pattern = (r"(b\.?\s?tech|b\.?\s?e\.?|m\.?\s?tech|m\.?\s?e\.?|b\.?\s?sc|m\.?\s?sc|"
               r"bca|mca|mba|ph\.?d|bachelor[^\n,]*|master[^\n,]*)")
    found = re.findall(pattern, text, flags=re.IGNORECASE)
    return sorted({f.strip().upper() for f in found})


def extract_experience_years(text):
    matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", text, flags=re.IGNORECASE)
    return max(map(float, matches)) if matches else 0.0


def extract_cgpa(text):
    match = re.search(r"(?:cgpa|gpa)[:\s]*([0-9]\.\d{1,2})", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


# ---------- Main parse function ----------
def parse_resume(text: str) -> dict:
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience_years": extract_experience_years(text),
        "cgpa": extract_cgpa(text),
    }


if __name__ == "__main__":
    from extractor import extract_text
    text = extract_text("data/sample_resumes/resume1.pdf")
    import json
    print(json.dumps(parse_resume(text), indent=2))