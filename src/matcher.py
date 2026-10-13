from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

from extractor import extract_text
from parser import parse_resume, extract_skills

# ---------- Weights (must add up to 1.0) ----------
WEIGHTS = {"skills": 0.5, "tfidf": 0.2, "semantic": 0.3}

# Load the embedding model once (slow to load, so do it lazily)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ---------- 1. Skill overlap ----------
def skill_match(resume_skills, jd_skills):
    resume_set, jd_set = set(resume_skills), set(jd_skills)
    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    score = len(matched) / len(jd_set) if jd_set else 0.0
    return score, matched, missing


# ---------- 2. TF-IDF similarity ----------
def tfidf_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([resume_text, jd_text])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


# ---------- 3. Semantic similarity (Sentence-BERT) ----------
def semantic_similarity(resume_text, jd_text):
    model = get_model()
    embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)
    score = float(util.cos_sim(embeddings[0], embeddings[1]))
    return max(0.0, score)


# ---------- Main matching function ----------
def match_resume_to_jd(resume_text: str, jd_text: str) -> dict:
    resume_skills = parse_resume(resume_text)["skills"]
    jd_skills = extract_skills(jd_text)

    skill_score, matched, missing = skill_match(resume_skills, jd_skills)
    tfidf_score = tfidf_similarity(resume_text, jd_text)
    semantic_score = semantic_similarity(resume_text, jd_text)

    final = (WEIGHTS["skills"] * skill_score
             + WEIGHTS["tfidf"] * tfidf_score
             + WEIGHTS["semantic"] * semantic_score)

    return {
        "final_score": round(final * 100, 1),
        "skill_score": round(skill_score * 100, 1),
        "tfidf_score": round(tfidf_score * 100, 1),
        "semantic_score": round(semantic_score * 100, 1),
        "jd_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing,
    }


if __name__ == "__main__":
    import json
    resume_text = extract_text("data/sample_resumes/resume1.pdf")
    jd_text = extract_text("data/sample_jds/jd1.txt")
    print(json.dumps(match_resume_to_jd(resume_text, jd_text), indent=2))