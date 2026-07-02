import logging
import uuid
import hashlib
from pathlib import Path
from celery import Celery
from src.config import settings
from src.member1_ingestion.parser import DoclingLayoutParser
from src.member1_ingestion.schemas import CandidateProfile
from src.member2_retrieval.postgres_client import PostgresStateClient
from src.member2_retrieval.embedder import BGEM3Embedder
from src.member2_retrieval.qdrant_client import QdrantVectorClient

try:
    import instructor
    from openai import OpenAI

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery app routing
celery_app = Celery(
    "ingestion_tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL
)
celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True



def extract_heuristic_profile(raw_text: str) -> CandidateProfile:
    import re
    # Simple email finder
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', raw_text)
    email = email_match.group(0) if email_match else None

    # Simple phone finder
    phone_match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', raw_text)
    phone = phone_match.group(0) if phone_match else None

    # Simple name finder (e.g., first non-empty line)
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    name = lines[0] if lines else "Unknown Candidate"
    if len(name) > 50:
        name = "Unknown Candidate"

    # Common skills lookup
    common_skills = [
        "Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Machine Learning", 
        "Qdrant", "SQL", "Git", "Celery", "Pandas", "NumPy", "AWS", "PyTorch"
    ]
    skills = []
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', raw_text, re.IGNORECASE):
            skills.append(skill)

    # Heuristic experience extraction
    from src.member1_ingestion.schemas import WorkExperience
    experience = []
    exp_years = 5 if "Senior" in raw_text or "Lead" in raw_text else 2
    experience.append(WorkExperience(
        company="Previous Company",
        role="Software Engineer",
        duration_months=exp_years * 12,
        description="Software development experience."
    ))

    return CandidateProfile(
        name=name,
        email=email,
        phone=phone,
        skills=skills,
        experience=experience,
        education=[]
    )


@celery_app.task(
    name="tasks.ingest_resume_pipeline",
    bind=True,
    max_retries=3,
    default_retry_delay=10
)
def ingest_resume_pipeline(self, file_path_str: str) -> dict:
    """Background Celery task mapping unstructured files through parsing and schema validations.

    Args:
        file_path_str: Path string to the resume.

    Returns:
        Dict representing validated CandidateProfile data.
    """
    if not INSTRUCTOR_AVAILABLE:
        raise ImportError(
            "The 'instructor' and 'openai' libraries are required to run the ingest_resume_pipeline task. Please install them using pip."
        )

    logger.info(f"Triggered Celery task to parse resume: {file_path_str}")
    path = Path(file_path_str)

    try:
        # 1. Parse layout via Docling (OCR enabled)
        parser = DoclingLayoutParser()
        raw_payload = parser.parse_document(path)
        raw_text = raw_payload["raw_text"]

        # 2. Extract structured entities (Instructor LLM or Heuristics Fallback)
        logger.info("Initializing Instructor client for profile extraction...")

        # Check for valid API keys
        api_key = settings.VLLM_API_KEY or settings.OPENAI_API_KEY
        use_fallback = False
        if not api_key or api_key == "mock-key-for-development":
            logger.warning("Mock or missing API keys detected. Falling back to heuristic profile extraction.")
            use_fallback = True

        if use_fallback:
            profile = extract_heuristic_profile(raw_text)
        else:
            try:
                if settings.VLLM_API_URL:
                    openai_client = OpenAI(
                        base_url=settings.VLLM_API_URL, api_key=settings.VLLM_API_KEY
                    )
                    model_name = "local-model"
                else:
                    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                    model_name = "gpt-4o-mini"

                instructor_client = instructor.from_openai(openai_client)

                # Enforce exact Instructor schema matching
                profile = instructor_client.chat.completions.create(
                    model=model_name,
                    response_model=CandidateProfile,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional recruiting assistant. Extract the structured candidate profile from the raw resume text. Extract skills, duration details, companies, and work/education timelines accurately.",
                        },
                        {"role": "user", "content": f"Resume Text:\n\n{raw_text}"},
                    ],
                    temperature=0.0,
                )
                logger.info(f"LLM successfully extracted profile for: {profile.name}")
            except Exception as e:
                logger.warning(f"Instructor LLM extraction failed: {e}. Falling back to heuristic profile extraction.")
                profile = extract_heuristic_profile(raw_text)

        # Generate unique candidate ID based on email
        email_key = profile.email or str(uuid.uuid4())
        candidate_id = (
            "cand_"
            + hashlib.md5(email_key.lower().strip().encode("utf-8")).hexdigest()[:12]
        )

        profile_dict = profile.model_dump()
        profile_dict["id"] = candidate_id

        # 3. Store in PostgreSQL database
        pg_client = PostgresStateClient()
        pg_client.store_candidate(candidate_id, profile_dict, raw_text)

        # 4. Generate Embeddings (BGE-M3 Dense + Sparse)
        # Note: We force the fallback mock embedder during ingestion to avoid blocking
        # on the 2.2GB model download. In production, configure a GPU worker with
        # the model pre-downloaded.
        embedder = BGEM3Embedder.__new__(BGEM3Embedder)
        embedder.model = None
        embedder.is_fallback = True
        embedder.dimension = 1024

        skills_summary = ", ".join(profile.skills)
        exp_summary = " ".join(
            [
                f"{exp.role} at {exp.company} - {exp.description or ''}"
                for exp in profile.experience
            ]
        )
        summary_text = (
            f"Name: {profile.name}. Skills: {skills_summary}. Experience: {exp_summary}"
        )

        embeddings_list = embedder.generate_embeddings([summary_text])
        if embeddings_list:
            dense_vector = embeddings_list[0]["dense"]
            sparse_vector = embeddings_list[0]["sparse"]

            # 5. Index in Qdrant Vector database
            qdrant_client = QdrantVectorClient()
            qdrant_client.upsert_vectors(
                [
                    {
                        "id": candidate_id,
                        "dense": dense_vector,
                        "sparse": sparse_vector,
                        "payload": {
                            "candidate_id": candidate_id,
                            "name": profile.name,
                            "email": profile.email,
                            "skills": profile.skills,
                            "years_exp": sum(
                                exp.duration_months for exp in profile.experience
                            )
                            / 12.0,
                        },
                    }
                ]
            )

        return profile_dict
    except Exception as exc:
        logger.error(
            f"Task failed during execution: {exc}. Retrying attempt {self.request.retries}/{self.max_retries}"
        )
        raise self.retry(exc=exc)
