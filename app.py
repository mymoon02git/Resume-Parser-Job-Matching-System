import os
import sys
import tempfile

import pandas as pd
import streamlit as st

# Make the src/ folder importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from extractor import extract_text
from parser import parse_resume
import matcher

st.set_page_config(page_title="Resume Parser & Job Matcher", page_icon="📄", layout="wide")


# ---------- Helpers ----------
def read_uploaded(file) -> str:
    """Save an uploaded file to a temp path, extract its text, then delete it."""
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getbuffer())
        path = tmp.name
    try:
        return extract_text(path)
    finally:
        os.remove(path)


# ---------- Sidebar: scoring weights ----------
st.sidebar.header("Scoring weights")
w_skills = st.sidebar.slider("Skill overlap", 0.0, 1.0, 0.5, 0.05)
w_tfidf = st.sidebar.slider("TF-IDF similarity", 0.0, 1.0, 0.2, 0.05)
w_sem = st.sidebar.slider("Semantic similarity", 0.0, 1.0, 0.3, 0.05)

total = w_skills + w_tfidf + w_sem
if total == 0:
    st.sidebar.error("At least one weight must be above 0.")
    st.stop()

# Normalise so the weights always add up to 1
matcher.WEIGHTS.update({
    "skills": w_skills / total,
    "tfidf": w_tfidf / total,
    "semantic": w_sem / total,
})
st.sidebar.caption("Weights are normalised automatically to add up to 100%.")


# ---------- Main page ----------
st.title("📄 Resume Parser & Job Matching System")
st.write("Upload resumes, paste a job description, and rank candidates by fit.")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Job description")
    jd_text = st.text_area("Paste the job description", height=280,
                           placeholder="Requirements: Python, SQL, machine learning ...")

with col_right:
    st.subheader("2. Resumes")
    files = st.file_uploader("Upload one or more resumes (PDF / DOCX / TXT)",
                             type=["pdf", "docx", "txt"], accept_multiple_files=True)

if st.button("Rank candidates", type="primary"):
    if not jd_text.strip():
        st.warning("Please paste a job description.")
        st.stop()
    if not files:
        st.warning("Please upload at least one resume.")
        st.stop()

    results = []
    progress = st.progress(0.0, text="Analysing resumes...")

    for i, file in enumerate(files):
        try:
            resume_text = read_uploaded(file)
            if not resume_text.strip():
                st.warning(f"{file.name}: no text found (scanned image PDF?). Skipped.")
                continue

            parsed = parse_resume(resume_text)
            score = matcher.match_resume_to_jd(resume_text, jd_text)
            results.append({"file": file.name, "parsed": parsed, "score": score})
        except Exception as e:
            st.error(f"{file.name}: could not be processed ({e})")

        progress.progress((i + 1) / len(files), text=f"Analysed {i + 1} of {len(files)}")

    progress.empty()

    if not results:
        st.stop()

    results.sort(key=lambda r: r["score"]["final_score"], reverse=True)

    # ----- Ranking table -----
    st.subheader("🏆 Ranking")
    table = pd.DataFrame([{
        "Rank": rank,
        "Candidate": r["parsed"]["name"],
        "File": r["file"],
        "Final score": r["score"]["final_score"],
        "Skill match %": r["score"]["skill_score"],
        "TF-IDF %": r["score"]["tfidf_score"],
        "Semantic %": r["score"]["semantic_score"],
        "Missing skills": len(r["score"]["missing_skills"]),
    } for rank, r in enumerate(results, start=1)])

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download ranking as CSV", table.to_csv(index=False),
                       file_name="candidate_ranking.csv", mime="text/csv")

    # ----- Per-candidate details -----
    st.subheader("🔍 Candidate details")
    for rank, r in enumerate(results, start=1):
        p, s = r["parsed"], r["score"]
        with st.expander(f"#{rank}  {p['name']}  —  {s['final_score']}%", expanded=(rank == 1)):
            c1, c2, c3 = st.columns(3)
            c1.metric("Final score", f"{s['final_score']}%")
            c2.metric("Skill match", f"{s['skill_score']}%")
            c3.metric("Semantic match", f"{s['semantic_score']}%")

            st.markdown(f"**Email:** {p['email']}  |  **Phone:** {p['phone']}")
            st.markdown(f"**Education:** {', '.join(p['education']) or 'Not found'}  |  "
                        f"**CGPA:** {p['cgpa'] or 'Not found'}")

            st.markdown("**✅ Matched skills**")
            st.success(", ".join(s["matched_skills"]) or "None")

            st.markdown("**❌ Missing skills**")
            if s["missing_skills"]:
                st.error(", ".join(s["missing_skills"]))
            else:
                st.success("The candidate has every skill the JD asks for.")