import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="NexusMatch Engine Dashboard", layout="wide")

st.title("🎯 NexusMatch Engine Analytics Panel")
st.subheader("Interactive Talent Matching, LTR Re-ranking, and Explainability Studio")

st.markdown("""
This interface interacts with the FastAPI backend orchestration layer. 
Upload resumes to ingest profiles or enter job requirements below to evaluate candidates in real-time.
""")

# Backend server address
API_URL = "http://localhost:8000"

# 1. Fetch Real-time Dashboard Metrics from PostgreSQL
try:
    metrics_response = requests.get(f"{API_URL}/metrics", timeout=3.0)
    if metrics_response.status_code == 200:
        metrics = metrics_response.json()
    else:
        metrics = {}
except Exception:
    metrics = {}

# Render Metrics Row
st.markdown("### 📊 Database Statistics (Real-time)")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric("Total Candidates Ingested", metrics.get("total_candidates", 0))
with m_col2:
    st.metric("Active Job Descriptions", metrics.get("total_jobs", 0))
with m_col3:
    st.metric(
        "Avg Experience (Years)", f"{metrics.get('avg_experience_years', 0.0)} yrs"
    )
with m_col4:
    top_skills_dict = metrics.get("top_skills", {})
    skills_str = (
        ", ".join([f"{k} ({v})" for k, v in top_skills_dict.items()])
        if top_skills_dict
        else "None"
    )
    st.metric("Top Skills in Repository", skills_str)

st.markdown("---")

# Sidebar controls
st.sidebar.header("📁 Ingestion Panel")
uploaded_file = st.sidebar.file_uploader(
    "Upload Candidate Resume (PDF / DOCX)", type=["pdf", "docx"]
)
if uploaded_file is not None:
    if st.sidebar.button("Process Resume"):
        # Make a real POST request to the FastAPI backend uploader
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        try:
            response = requests.post(
                f"{API_URL}/ingest/resume", files=files, timeout=30.0
            )
            if response.status_code == 200:
                data = response.json()
                st.sidebar.success(f"Resume queued! Task ID: {data.get('task_id')}")
            else:
                st.sidebar.error(f"Ingestion failed: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error connecting to API server: {e}")

# Main dashboard columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Job Requirements")
    job_id = st.text_input("Job ID", "job_data_science_01")
    job_title = st.text_input("Target Job Title", "Senior Data Scientist")
    jd_text = st.text_area(
        "Unstructured Job Description",
        "We are looking for a Senior Data Scientist with 5+ years of experience in Python, ML pipelines, and vector databases like Qdrant.",
    )
    skills_required = st.text_input(
        "Required Skills (Comma separated)", "Python, Machine Learning, Qdrant"
    )
    min_exp = st.slider("Min Years of Experience", 0, 15, 5)

with col2:
    st.header("🏆 Matched Candidates")
    if st.button("Run Match Engine Execution Loop", type="primary"):
        skills_list = [s.strip() for s in skills_required.split(",")]

        st.write("🔍 Running BGE-M3 Dense/Sparse embedding generations...")
        st.write("🛰️ Querying Qdrant index vector partitions...")
        st.write("⚡ Computing LambdaMART feature engineering matrices...")

        # Make a real POST request to match candidates
        payload = {
            "job_description_id": job_id,
            "job_text": jd_text,
            "required_skills": skills_list,
            "min_experience_years": min_exp,
            "top_k": 5,
        }

        try:
            response = requests.post(f"{API_URL}/match", json=payload, timeout=10.0)
            if response.status_code == 200:
                results = response.json().get("results", [])

                if not results:
                    st.warning(
                        "No matching candidates found in the vector index. Upload resumes first to build the database!"
                    )

                for res in results:
                    with st.container():
                        st.write(
                            f"### Rank {res.get('listwise_rank', 'N/A')}: {res.get('name')} ({res.get('id')})"
                        )
                        st.write(
                            f"**Final Rerank Score:** `{res.get('ltr_score', 0.0):.4f}` (Initial Similarity: `{res.get('initial_score', 0.0):.4f}`)"
                        )
                        st.write(
                            f"**LLM Explainability Rationale:** {res.get('llm_rationale')}"
                        )

                        # Render actual feature importance charts (SHAP)
                        shap_values = res.get("shap_values", {})
                        if shap_values:
                            shap_df = pd.DataFrame(
                                list(shap_values.items()),
                                columns=["Feature", "Impact (SHAP)"],
                            )
                            st.bar_chart(shap_df.set_index("Feature"))
                        st.markdown("---")
            else:
                st.error(f"Match query failed: {response.text}")
        except Exception as e:
            st.error(f"Error querying match API: {e}")
