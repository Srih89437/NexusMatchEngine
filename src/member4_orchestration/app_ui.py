import streamlit as st
import pandas as pd
import requests
from src.config import settings

st.set_page_config(page_title="Analysis Dashboard", layout="wide")

st.title("🎯 Analysis Dashboard")
st.subheader("Interactive Talent Matching, LTR Re-ranking, and Explainability Studio")

st.markdown(f"""
This interface interacts with the FastAPI backend orchestration layer. 
Upload resumes to ingest profiles or enter job requirements below to evaluate candidates in real-time.
""")

# Backend server address
API_URL = "http://localhost:8000"

# Initialize session state for tracking resume upload
if "resumes_uploaded" not in st.session_state:
    st.session_state["resumes_uploaded"] = False

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

# Initialize session state for match results
if "match_results" not in st.session_state:
    st.session_state["match_results"] = None
if "match_job_id" not in st.session_state:
    st.session_state["match_job_id"] = None

# Sidebar controls
st.sidebar.header("📁 Ingestion Panel")
clear_db = st.sidebar.checkbox("Clear existing database cache", value=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload Candidate Resume (PDF / DOCX)", type=["pdf", "docx"], accept_multiple_files=True
)
if uploaded_files:
    if st.sidebar.button("Process Resumes"):
        if clear_db:
            st.session_state["match_results"] = None
            st.session_state["match_job_id"] = None
            try:
                clear_res = requests.post(f"{API_URL}/clear", timeout=10.0)
                if clear_res.status_code == 200:
                    st.sidebar.info("🧹 Existing database cleared successfully.")
                else:
                    st.sidebar.error("⚠️ Failed to clear database cache.")
            except Exception as e:
                st.sidebar.error(f"Error calling clear API: {e}")

        total_files = len(uploaded_files)
        status_box = st.sidebar.empty()
        prog_bar = st.sidebar.progress(0, text=f"⏳ Ingesting resumes… 0/{total_files}")

        success_count = 0
        failed_count = 0

        for idx, uploaded_file in enumerate(uploaded_files):
            pct = idx / total_files
            prog_bar.progress(pct, text=f"⏳ Ingesting {uploaded_file.name} ({idx+1}/{total_files})")
            
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            try:
                response = requests.post(
                    f"{API_URL}/ingest/resume",
                    files=files,
                    timeout=125,
                )
                if response.status_code == 200:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1

        prog_bar.empty()
        if failed_count == 0:
            status_box.success(f"✅ All {success_count} resumes processed successfully!")
            st.session_state["resumes_uploaded"] = True
            st.rerun()
        else:
            status_box.warning(f"Ingestion completed: {success_count} successful, {failed_count} failed.")
            if success_count > 0:
                st.session_state["resumes_uploaded"] = True
                st.rerun()

# Main dashboard columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Job Requirements")
    job_id = st.text_input("Job ID", value="", placeholder="e.g. job_data_science_01")
    job_title = st.text_input(
        "Target Job Title", value="", placeholder="e.g. Senior Data Scientist"
    )
    jd_text = st.text_area(
        "Unstructured Job Description",
        value="",
        placeholder="Enter job description details here...",
    )
    skills_required = st.text_input(
        "Required Skills (Comma separated)",
        value="",
        placeholder="e.g. Python, Machine Learning, Qdrant",
    )
    min_exp = st.slider("Min Years of Experience", 0, 15, 5)

with col2:
    total_candidates = metrics.get("total_candidates", 0)
    
    if total_candidates > 0:
        st.header("🏆 Matched Candidates")
    
    if st.button("Run Match Engine Execution Loop", type="primary"):
        # Validate that a resume has been uploaded by checking the metrics candidates count
        if total_candidates == 0:
            st.error("Please upload a resume before running the Match Engine.")
            st.stop()
 
        if not job_id.strip():
            st.error("⚠️ Please enter a valid Job ID before running the Match Engine.")
            st.stop()
        if not jd_text.strip():
            st.error("⚠️ Please enter the Job Description before running the Match Engine.")
            st.stop()
 
        skills_list = [s.strip() for s in skills_required.split(",")] if skills_required.strip() else []
 
        # Show processing logs/animations ONLY during real execution
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
            with st.spinner("Executing matches..."):
                response = requests.post(f"{API_URL}/match", json=payload, timeout=20.0)
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if not results:
                        st.session_state["match_results"] = None
                        st.session_state["match_job_id"] = None
                        st.warning("No matching candidates found in the vector index. Upload resumes first to build the database!")
                    else:
                        st.session_state["match_results"] = results
                        st.session_state["match_job_id"] = job_id
                        st.rerun()
                else:
                    st.session_state["match_results"] = None
                    st.session_state["match_job_id"] = None
                    try:
                        error_detail = response.json().get("detail", response.text)
                    except Exception:
                        error_detail = response.text
                    st.error(f"⚠️ Match query failed: {error_detail}")
        except Exception as e:
            st.session_state["match_results"] = None
            st.session_state["match_job_id"] = None
            st.error(f"Error querying match API: {e}")
 
    # Render persistent results if they exist in session state
    has_results = (st.session_state["match_results"] is not None and st.session_state["match_job_id"] == job_id)
 
    if has_results:
        results = st.session_state["match_results"]
        # Candidate Score Comparison Chart
        st.write("### 📈 Candidate Score Comparison")
        chart_data = pd.DataFrame([
            {"Candidate": f"{res.get('name')} ({res.get('id')})", "Final Score": res.get("ltr_score", 0.0)}
            for res in results
        ])
        st.bar_chart(chart_data.set_index("Candidate"))
 
        # Export buttons shown only after the graph is successfully rendered and visible
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            try:
                xlsx_res = requests.get(
                    f"{API_URL}/api/v1/results/{job_id}/download", timeout=10.0
                )
                if xlsx_res.status_code == 200:
                    st.download_button(
                        label="📥 Download Match Results (.xlsx)",
                        data=xlsx_res.content,
                        file_name=f"match_results_{job_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="xlsx_active"
                    )
                else:
                    st.error(f"Failed to generate Excel download: {xlsx_res.text}")
            except Exception as e:
                st.error(f"Error calling download API: {e}")
 
        with col_dl2:
            try:
                csv_res = requests.get(
                    f"{API_URL}/api/v1/results/{job_id}/download_csv", timeout=10.0
                )
                if csv_res.status_code == 200:
                    st.download_button(
                        label="📥 Download Match Results (CSV)",
                        data=csv_res.content,
                        file_name=f"match_results_{job_id}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="csv_active"
                    )
                else:
                    st.error(f"Failed to generate CSV download: {csv_res.text}")
            except Exception as e:
                st.error(f"Error calling CSV download API: {e}")
 
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
        st.info("No matching data available. Upload a resume and run the Match Engine to generate results.")
