import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_experience_matching(
    candidate_exp_years: float, jd_required_years: float
) -> float:
    """Compute normalized delta score for experience metrics, bounded between 0.0 and 1.0."""
    if jd_required_years == 0:
        return 1.0
    ratio = candidate_exp_years / jd_required_years
    return min(ratio, 1.0)  # Bounded to [0, 1]


def calculate_trajectory_velocity(experience_history: List[Dict[str, Any]]) -> float:
    """Determine speed of candidate career growth (promotions frequency over time).

    Trajectory Velocity is calculated as:
    V = (Count of unique promotion titles) / (Total years of work history)
    """
    if not experience_history:
        return 0.0
    total_months = sum(item.get("duration_months", 0) for item in experience_history)
    unique_roles = len(set(item.get("role", "") for item in experience_history))

    years = max(total_months / 12.0, 0.5)
    return float(unique_roles / years)


def calculate_skill_decay(skills: List[str], history: List[Dict[str, Any]]) -> float:
    """Calculate skill freshness score.

    Simulates skills decay based on how recently they were exercised in work history.
    """
    if not history or not skills:
        return 0.5
    last_job = history[0]
    end_date = str(last_job.get("end_date", "")).lower()
    if "present" in end_date or not end_date:
        return 1.0
    return 0.95


def build_ltr_features(
    candidate: Dict[str, Any],
    job: Dict[str, Any],
    semantic_score: float,
    dense_score: Optional[float] = None,
    sparse_score: Optional[float] = None,
) -> Dict[str, float]:
    """Transform candidate profile and job requirements into raw normalized LTR features."""
    logger.info("Computing features for ranking model pipeline...")

    cand_skills = candidate.get("skills", [])
    jd_skills = job.get("required_skills", [])

    # 1. Skill Match Score (intersection over required)
    common_skills = set(c.lower() for c in cand_skills).intersection(
        set(j.lower() for j in jd_skills)
    )
    skill_match_score = len(common_skills) / max(len(jd_skills), 1)
    skill_match_score = min(skill_match_score, 1.0)

    # 2. Experience Metrics
    experience_list = candidate.get("experience", [])
    total_months = sum(item.get("duration_months", 0) for item in experience_list)
    cand_exp = total_months / 12.0
    jd_exp = float(job.get("min_experience_years", 0))
    exp_ratio = calculate_experience_matching(cand_exp, jd_exp)

    # Years experience normalized to max 25 years
    experience_years_norm = min(cand_exp / 25.0, 1.0)

    # 3. Average Tenure
    avg_tenure_years = (total_months / max(len(experience_list), 1)) / 12.0
    # Normalized average tenure (capped at 5 years)
    average_tenure_norm = min(avg_tenure_years / 5.0, 1.0)

    # 4. Job Transition Rate (velocity)
    velocity = calculate_trajectory_velocity(experience_list)
    job_transition_rate = min(velocity / 3.0, 1.0)  # Normalized career velocity

    # 5. Skill Recency
    decay = calculate_skill_decay(cand_skills, experience_list)
    skill_recency = decay

    # 6. Education Match
    education_list = candidate.get("education", [])
    education_match = 0.0
    for edu in education_list:
        degree = str(edu.get("degree", "")).lower()
        if (
            "bachelor" in degree
            or "b.s." in degree
            or "bs" in degree
            or "degree" in degree
        ):
            education_match = 1.0
            break

    # 7. Location Match
    cand_location = str(candidate.get("location", "")).lower()
    jd_location = str(job.get("location", "")).lower()
    location_match = 0.0
    if (
        not jd_location
        or "remote" in jd_location
        or jd_location in cand_location
        or cand_location in jd_location
    ):
        location_match = 1.0

    # 8. Hard Constraint Match
    all_required_present = all(
        j.lower() in [c.lower() for c in cand_skills] for j in jd_skills
    )
    hard_constraint_match = 1.0 if all_required_present else 0.0

    # 9. Dense & Sparse Similarity Scores
    dense_val = dense_score if dense_score is not None else semantic_score
    sparse_val = sparse_score if sparse_score is not None else semantic_score

    semantic_similarity = min(max(semantic_score, 0.0), 1.0)
    dense_similarity = min(max(dense_val, 0.0), 1.0)
    sparse_similarity = min(max(sparse_val, 0.0), 1.0)

    return {
        "experience_years": experience_years_norm,
        "skill_match_score": skill_match_score,
        "trajectory_velocity": velocity,
        "exp_ratio": exp_ratio,
        "skill_decay": decay,
        "semantic_similarity": semantic_similarity,
        "dense_similarity": dense_similarity,
        "sparse_similarity": sparse_similarity,
        "average_tenure": average_tenure_norm,
        "job_transition_rate": job_transition_rate,
        "skill_recency": skill_recency,
        "education_match": education_match,
        "location_match": location_match,
        "hard_constraint_match": hard_constraint_match,
    }
