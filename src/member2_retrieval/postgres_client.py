import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class CandidateModel(Base):
    __tablename__ = "candidates"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=True)
    phone = Column(String(50), nullable=True)
    skills = Column(JSON, nullable=True)  # List of strings
    experience = Column(JSON, nullable=True)  # List of WorkExperience dicts
    education = Column(JSON, nullable=True)  # List of Education dicts
    raw_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class JobDescriptionModel(Base):
    __tablename__ = "job_descriptions"

    id = Column(String(50), primary_key=True)
    title = Column(String(150), nullable=False)
    department = Column(String(100), nullable=True)
    required_skills = Column(JSON, nullable=True)  # List of strings
    preferred_skills = Column(JSON, nullable=True)  # List of strings
    min_experience_years = Column(Integer, default=0)
    location = Column(String(100), nullable=True)
    visa_requirement = Column(String(100), nullable=True)
    full_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PostgresStateClient:
    """Relational storage tier for caching structured entities and logs."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.DATABASE_URL

        try:
            logger.info("Initializing connection engine...")
            self.engine = create_engine(self.db_url, pool_pre_ping=True)
            # Test connection
            with self.engine.connect() as conn:
                pass
            logger.info("Connected to PostgreSQL successfully.")
        except Exception as e:
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[2]
            db_path = project_root / "data" / "nexusmatch.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"PostgreSQL connection to {self.db_url} failed: {e}. Falling back to file-based SQLite database at {db_path}."
            )
            self.db_url = f"sqlite:///{db_path}"
            self.engine = create_engine(self.db_url)

        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)

        self.create_tables()

    def create_tables(self):
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Postgres database schemas initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to create schema tables: {e}")

    def store_candidate(
        self,
        candidate_id: str,
        profile_data: Dict[str, Any],
        raw_text: Optional[str] = None,
    ) -> bool:
        """Store structured Candidate profile in permanent relational state."""
        session = self.Session()
        try:
            candidate = session.query(CandidateModel).filter_by(id=candidate_id).first()
            if not candidate:
                candidate = CandidateModel(id=candidate_id)
                session.add(candidate)

            candidate.name = profile_data.get("name", "Unknown")
            candidate.email = profile_data.get("email")
            candidate.phone = profile_data.get("phone")
            candidate.skills = profile_data.get("skills", [])
            candidate.experience = profile_data.get("experience", [])
            candidate.education = profile_data.get("education", [])
            candidate.raw_text = raw_text or profile_data.get("raw_text", "")
            candidate.created_at = datetime.utcnow()

            session.commit()
            logger.info(f"Successfully stored candidate profile: {candidate_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error executing store_candidate for {candidate_id}: {e}")
            return False
        finally:
            self.Session.remove()

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve candidate profile details from tables."""
        session = self.Session()
        try:
            candidate = session.query(CandidateModel).filter_by(id=candidate_id).first()
            if not candidate:
                return None
            return {
                "id": candidate.id,
                "name": candidate.name,
                "email": candidate.email,
                "phone": candidate.phone,
                "skills": candidate.skills,
                "experience": candidate.experience,
                "education": candidate.education,
                "raw_text": candidate.raw_text,
                "created_at": (
                    candidate.created_at.isoformat() if candidate.created_at else None
                ),
            }
        except Exception as e:
            logger.error(f"Error fetching candidate {candidate_id}: {e}")
            return None
        finally:
            self.Session.remove()

    def store_job_description(self, job_id: str, jd_data: Dict[str, Any]) -> bool:
        """Store structured Job Description metadata in permanent relational state."""
        session = self.Session()
        try:
            jd = session.query(JobDescriptionModel).filter_by(id=job_id).first()
            if not jd:
                jd = JobDescriptionModel(id=job_id)
                session.add(jd)

            jd.title = jd_data.get("title", "Unknown Job Title")
            jd.department = jd_data.get("department")
            jd.required_skills = jd_data.get("required_skills", [])
            jd.preferred_skills = jd_data.get("preferred_skills", [])
            jd.min_experience_years = jd_data.get("min_experience_years", 0)
            jd.location = jd_data.get("location")
            jd.visa_requirement = jd_data.get("visa_requirement")
            jd.full_text = jd_data.get("full_text", "")
            jd.created_at = datetime.utcnow()

            session.commit()
            logger.info(f"Successfully stored job description: {job_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error executing store_job_description for {job_id}: {e}")
            return False
        finally:
            self.Session.remove()

    def get_job_description(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job description details from tables."""
        session = self.Session()
        try:
            jd = session.query(JobDescriptionModel).filter_by(id=job_id).first()
            if not jd:
                return None
            return {
                "id": jd.id,
                "title": jd.title,
                "department": jd.department,
                "required_skills": jd.required_skills,
                "preferred_skills": jd.preferred_skills,
                "min_experience_years": jd.min_experience_years,
                "location": jd.location,
                "visa_requirement": jd.visa_requirement,
                "full_text": jd.full_text,
                "created_at": jd.created_at.isoformat() if jd.created_at else None,
            }
        except Exception as e:
            logger.error(f"Error fetching job description {job_id}: {e}")
            return None
        finally:
            self.Session.remove()

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Aggregate relational stats for the recruiter dashboard."""
        session = self.Session()
        try:
            total_candidates = session.query(CandidateModel).count()
            total_jobs = session.query(JobDescriptionModel).count()

            candidates = session.query(CandidateModel).all()
            total_exp_years = 0.0
            all_skills = []

            for c in candidates:
                exp_list = c.experience or []
                months = sum(e.get("duration_months", 0) for e in exp_list)
                total_exp_years += months / 12.0
                all_skills.extend(c.skills or [])

            avg_exp = (
                float(total_exp_years / total_candidates)
                if total_candidates > 0
                else 0.0
            )

            skill_counts = {}
            for s in all_skills:
                skill_counts[s] = skill_counts.get(s, 0) + 1
            top_skills = dict(
                sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            )

            return {
                "total_candidates": total_candidates,
                "total_jobs": total_jobs,
                "avg_experience_years": round(avg_exp, 1),
                "top_skills": top_skills,
            }
        except Exception as e:
            logger.error(f"Failed to query dashboard metrics: {e}")
            return {
                "total_candidates": 0,
                "total_jobs": 0,
                "avg_experience_years": 0.0,
                "top_skills": {},
            }
        finally:
            self.Session.remove()
