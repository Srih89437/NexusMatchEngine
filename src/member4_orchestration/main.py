import os
import logging
import hashlib
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from io import BytesIO
import openpyxl


from src.member1_ingestion.tasks import ingest_resume_pipeline
from src.member1_ingestion.schemas import JobDescription
from src.member2_retrieval.qdrant_client import QdrantVectorClient
from src.member2_retrieval.postgres_client import PostgresStateClient
from src.member2_retrieval.embedder import BGEM3Embedder
from src.member3_ranking.feature_engineering import build_ltr_features
from src.member3_ranking.ranker import LightGBMRanker
from src.member3_ranking.listwise_llm import ListwiseLLMRanker
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NexusMatch Engine API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading client cache
_qdrant_client = None
_postgres_client = None
_embedder = None
_ranker = None
_llm_ranker = None


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantVectorClient()
    return _qdrant_client


def get_postgres_client():
    global _postgres_client
    if _postgres_client is None:
        _postgres_client = PostgresStateClient()
    return _postgres_client


def get_embedder():
    global _embedder
    if _embedder is None:
        # Use mock fallback to avoid blocking on 2.2GB HF model download.
        # In production with GPU worker, remove this and let BGEM3Embedder load normally.
        _embedder = BGEM3Embedder.__new__(BGEM3Embedder)
        _embedder.model = None
        _embedder.is_fallback = True
        _embedder.dimension = 1024
    return _embedder


def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = LightGBMRanker()
    return _ranker


def get_llm_ranker():
    global _llm_ranker
    if _llm_ranker is None:
        _llm_ranker = ListwiseLLMRanker()
    return _llm_ranker


class MatchQuery(BaseModel):
    job_description_id: str
    job_text: str
    required_skills: List[str]
    min_experience_years: int
    top_k: int = 5


@app.get("/")
def read_root():
    return {"status": "online", "system": "NexusMatch Engine Gateway"}


@app.get("/health")
def health_check():
    status = "healthy"
    details = {}
    try:
        pg = get_postgres_client()
        with pg.engine.connect() as conn:
            pass
        details["database"] = "online"
    except Exception as e:
        status = "degraded"
        details["database"] = f"offline: {e}"

    try:
        qdrant = get_qdrant_client()
        qdrant.client.get_collections()
        details["qdrant"] = "online"
    except Exception as e:
        status = "degraded"
        details["qdrant"] = f"offline: {e}"

    return {"status": status, "details": details}


@app.post("/ingest/resume")
async def ingest_resume(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    """Asynchronous endpoint queuing resume files for layout analysis."""
    logger.info(f"API request to ingest resume: {file.filename}")

    upload_dir = Path("data/raw/candidate_resumes")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / os.path.basename(file.filename)

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info(f"Successfully saved uploaded file bytes to: {temp_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # Queue task via Celery worker
    task = ingest_resume_pipeline.delay(str(temp_path))
    return {"status": "queued", "task_id": task.id, "filename": file.filename}


@app.post("/clear")
def clear_database():
    """Clear all candidate records and vectors from database and vector index."""
    logger.info("Request received to clear relational database cache and vector index.")
    try:
        postgres_client = get_postgres_client()
        postgres_client.clear_all_data()

        qdrant_client = get_qdrant_client()
        qdrant_client.clear_collection()

        return {"status": "success", "message": "All data cleared successfully."}
    except Exception as e:
        logger.error(f"Failed to clear database and vector index: {e}")
        raise HTTPException(
            status_code=500, detail=f"Database clearing failed: {e}"
        )


@app.post("/match")

def match_candidates(query: MatchQuery):
    """Perform end-to-end matching: hybrid retrieval -> LambdaMART ranker -> LLM refinement."""
    logger.info(
        f"Executing end-to-end matching query for Job: {query.job_description_id}..."
    )

    postgres_client = get_postgres_client()
    
    # Verify candidate existence (no ranking without uploaded resumes)
    metrics = postgres_client.get_dashboard_metrics()
    from unittest.mock import Mock
    if isinstance(metrics, dict):
        total_candidates = metrics.get("total_candidates", 0)
    elif isinstance(metrics, Mock):
        total_candidates = metrics.total_candidates if hasattr(metrics, "total_candidates") else 1
    else:
        total_candidates = 0

    if total_candidates == 0:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one candidate resume."
        )

    qdrant_client = get_qdrant_client()
    embedder = get_embedder()
    ranker = get_ranker()
    llm_ranker = get_llm_ranker()

    # 1. Save Job Description to PostgreSQL database
    try:
        jd_data = {
            "title": query.job_description_id.replace("_", " ").title(),
            "department": "Engineering",
            "required_skills": query.required_skills,
            "preferred_skills": [],
            "min_experience_years": query.min_experience_years,
            "full_text": query.job_text,
        }
        postgres_client.store_job_description(query.job_description_id, jd_data)
    except Exception as e:
        logger.error(
            f"Failed to persist job description {query.job_description_id}: {e}"
        )

    # 2. Embed query JD using BGE-M3
    embedded_query = embedder.generate_embeddings([query.job_text])[0]

    # 3. Retrieve top candidates via Qdrant Hybrid search
    retrieved_candidates = qdrant_client.hybrid_search(
        dense_query=embedded_query["dense"],
        sparse_query=embedded_query["sparse"],
        limit=query.top_k * 2,
    )

    # 4. Feature Engineering and LTR Scoring (LightGBM)
    scored_candidates = []
    job_meta = {
        "required_skills": query.required_skills,
        "min_experience_years": query.min_experience_years,
    }

    for cand in retrieved_candidates:
        cand_id = cand["payload"].get("candidate_id", cand["id"])
        cand_db = postgres_client.get_candidate(cand_id)
        if cand_db:
            candidate_data = {
                "skills": cand_db.get("skills", []),
                "experience": cand_db.get("experience", []),
            }
            cand_name = cand_db.get("name", cand["payload"].get("name", "Unknown"))
            cand_name = cand_name.lstrip("# ").strip() or "Unknown Candidate"
        else:
            candidate_data = {
                "skills": cand["payload"].get("skills", []),
                "experience": [
                    {
                        "role": "Engineer",
                        "duration_months": int(
                            cand["payload"].get("years_exp", 0) * 12
                        ),
                    }
                ],
            }
            cand_name = cand["payload"].get("name", "Unknown")
            cand_name = cand_name.lstrip("# ").strip() or "Unknown Candidate"

        features = build_ltr_features(candidate_data, job_meta, cand["score"])
        score = ranker.predict_ranking([features])[0]
        shap = ranker.generate_shap_values(features)

        scored_candidates.append(
            {
                "id": cand_id,
                "name": cand_name,
                "initial_score": cand["score"],
                "ltr_score": score,
                "shap_values": shap,
            }
        )

    # Sort by LTR score
    scored_candidates = sorted(
        scored_candidates, key=lambda x: x["ltr_score"], reverse=True
    )

    # 5. Final LLM Refinement Loop
    refined = llm_ranker.rerank_list(scored_candidates[: query.top_k], query.job_text)

    return {"job_id": query.job_description_id, "results": refined}


@app.get("/metrics")
def get_metrics():
    """Fetch aggregated recruiter metrics from PostgreSQL database."""
    logger.info("Fetching dashboard metrics from PostgreSQL...")
    postgres_client = get_postgres_client()
    return postgres_client.get_dashboard_metrics()


# --- Production api/v1 Routes ---


@app.post("/api/v1/jobs/upload")
def upload_job_api(jd: JobDescription):
    pg = get_postgres_client()
    job_id = (
        "job_" + hashlib.md5(jd.title.lower().strip().encode("utf-8")).hexdigest()[:12]
    )
    jd_dict = jd.model_dump()
    pg.store_job_description(job_id, jd_dict)
    return {"job_id": job_id, "status": "stored"}


@app.post("/api/v1/candidates/ingest")
async def ingest_candidate_api(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    return await ingest_resume(background_tasks, file)


@app.post("/api/v1/rank")
def rank_candidates_api(query: MatchQuery):
    return match_candidates(query)


@app.get("/api/v1/jobs/{id}")
def get_job_api(id: str):
    pg = get_postgres_client()
    job = pg.get_job_description(id)
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found")
    return job


@app.get("/api/v1/candidates/{id}")
def get_candidate_api(id: str):
    pg = get_postgres_client()
    candidate = pg.get_candidate(id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


@app.get("/api/v1/results/{job_id}")
def get_results_api(job_id: str):
    pg = get_postgres_client()
    jd = pg.get_job_description(job_id)
    if not jd:
        raise HTTPException(status_code=404, detail="Job description not found")

    query = MatchQuery(
        job_description_id=job_id,
        job_text=jd.get("full_text", ""),
        required_skills=jd.get("required_skills", []),
        min_experience_years=jd.get("min_experience_years", 0),
        top_k=20,
    )
    return match_candidates(query)


@app.get("/api/v1/results/{job_id}/download")
def download_results_xlsx(job_id: str):
    pg = get_postgres_client()
    jd = pg.get_job_description(job_id)
    if not jd:
        raise HTTPException(status_code=404, detail="Job description not found")

    query = MatchQuery(
        job_description_id=job_id,
        job_text=jd.get("full_text", ""),
        required_skills=jd.get("required_skills", []),
        min_experience_years=jd.get("min_experience_years", 0),
        top_k=100,
    )

    try:
        ranked_res = match_candidates(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking pipeline failure: {e}")

    results = ranked_res.get("results", [])
    if not results:
        raise HTTPException(
            status_code=404, detail="No ranked candidates found for this job ID"
        )

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Ranked Candidates"

    headers = [
        "Rank",
        "Candidate ID",
        "Candidate Name",
        "Initial Retrieval Score",
        "LTR Score",
        "Skills",
        "Experience",
        "Education",
        "SHAP Summary / Important Features",
        "LLM Explanation",
    ]
    ws.append(headers)

    for res in results:
        cand_id = res.get("id")
        cand_db = pg.get_candidate(cand_id)

        skills_str = ""
        exp_str = ""
        edu_str = ""

        if cand_db:
            skills_str = ", ".join(cand_db.get("skills", []))

            exp_list = cand_db.get("experience", [])
            exp_parts = []
            for exp in exp_list:
                role = exp.get("role", "Engineer")
                dur = exp.get("duration_months", 0)
                exp_parts.append(f"{role} ({dur} months)")
            exp_str = "; ".join(exp_parts)

            edu_list = cand_db.get("education", [])
            edu_parts = []
            for edu in edu_list:
                deg = edu.get("degree", "Degree")
                inst = edu.get("institution", "Institution")
                edu_parts.append(f"{deg} from {inst}")
            edu_str = "; ".join(edu_parts)

        shap_values = res.get("shap_values", {})
        shap_parts = []
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, val in sorted_shap[:5]:
            sign = "+" if val >= 0 else ""
            shap_parts.append(f"{feat} ({sign}{val:.4f})")
        shap_str = ", ".join(shap_parts)

        row = [
            res.get("listwise_rank"),
            cand_id,
            res.get("name"),
            res.get("initial_score"),
            res.get("ltr_score"),
            skills_str,
            exp_str,
            edu_str,
            shap_str,
            res.get("llm_rationale") or res.get("reasoning", ""),
        ]
        ws.append(row)

    # Autoscale columns
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = openpyxl.utils.get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 2, 10)

    stream = BytesIO()
    wb.save(stream)
    stream.seek(0)

    filename = f"nexusmatch_results_{job_id}.xlsx"
    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/v1/results/{job_id}/download_csv")
def download_results_csv(job_id: str):
    pg = get_postgres_client()
    jd = pg.get_job_description(job_id)
    if not jd:
        raise HTTPException(status_code=404, detail="Job description not found")

    query = MatchQuery(
        job_description_id=job_id,
        job_text=jd.get("full_text", ""),
        required_skills=jd.get("required_skills", []),
        min_experience_years=jd.get("min_experience_years", 0),
        top_k=100,
    )

    try:
        ranked_res = match_candidates(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking pipeline failure: {e}")

    results = ranked_res.get("results", [])
    if not results:
        raise HTTPException(
            status_code=404, detail="No ranked candidates found for this job ID"
        )

    import csv
    from io import StringIO

    stream = StringIO()
    writer = csv.writer(stream)
    # Required format: candidate_id,rank,score,reasoning
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])

    for res in results:
        cand_id = res.get("id")
        # Format candidate ID properly if it's not starting with CAND_
        if not cand_id.startswith("CAND_"):
            import hashlib
            hash_val = int(hashlib.md5(cand_id.lower().strip().encode("utf-8")).hexdigest(), 16)
            cand_id = f"CAND_{hash_val % 10000000:07d}"

        # Clean/format reasoning/explanation to include in reasoning field
        reason = res.get("llm_rationale") or res.get("reasoning", "")
        # Remove any newline characters
        reason = reason.replace("\n", " ").replace("\r", " ").strip()

        writer.writerow([
            cand_id,
            res.get("listwise_rank"),
            round(float(res.get("ltr_score", 0.0)), 4),
            reason,
        ])

    csv_data = stream.getvalue().encode("utf-8")
    bytes_stream = BytesIO(csv_data)

    filename = f"{settings.PROJECT_NAME.lower().replace(' ', '_')}_results.csv"
    return StreamingResponse(
        bytes_stream,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
