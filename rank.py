import json
import os
import csv
import argparse
import gzip
from datetime import datetime


def parse_date(date_str):
    if not date_str or date_str == "None":
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def evaluate_candidate(cand):
    cid = cand.get("candidate_id")
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})

    # 1. Honeypot check
    is_honeypot = False

    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months") == 0:
            is_honeypot = True

    for idx, job in enumerate(history):
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        if not end:
            end = datetime(2026, 7, 2)

        if start and end:
            expected_months = round((end - start).days / 30.43)
            duration = job.get("duration_months", 0)
            if abs(expected_months - duration) > 3:
                is_honeypot = True

    if is_honeypot:
        return None

    # 2. Exclude consulting-only
    consulting_firms = {
        "tcs",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini",
        "hcl",
        "tech mahindra",
        "l&t",
        "lnt",
        "mindtree",
        "tata consultancy services",
    }
    all_consulting = True
    if not history:
        all_consulting = False
    for job in history:
        comp = str(job.get("company", "")).lower()
        if not any(firm in comp for firm in consulting_firms):
            all_consulting = False
            break
    if all_consulting:
        return None

    # 3. Exclude pure academic/research
    all_academic = True
    if not history:
        all_academic = False
    for job in history:
        title = str(job.get("title", "")).lower()
        if not any(
            keyword in title
            for keyword in [
                "researcher",
                "research assistant",
                "postdoc",
                "academic",
                "phd",
            ]
        ):
            all_academic = False
            break
    if all_academic:
        return None

    # 4. Exclude CV/Speech/Robotics only
    cv_keywords = {
        "computer vision",
        "speech recognition",
        "robotics",
        "ros",
        "image classification",
        "object detection",
        "yolo",
    }
    nlp_keywords = {
        "nlp",
        "natural language processing",
        "rag",
        "embeddings",
        "vector database",
        "qdrant",
        "pinecone",
        "milvus",
        "weaviate",
        "elasticsearch",
        "opensearch",
        "search",
        "information retrieval",
        "transformer",
        "llm",
        "fine-tuning",
        "bert",
        "gpt",
    }

    has_cv = False
    has_nlp = False
    for s in skills:
        name = str(s.get("name", "")).lower()
        if any(kw in name for kw in cv_keywords):
            has_cv = True
        if any(kw in name for kw in nlp_keywords):
            has_nlp = True

    if has_cv and not has_nlp:
        return None

    # 5. Experience qualification
    exp_years = profile.get("years_of_experience", 0.0)
    if exp_years < 4.0 or exp_years > 15.0:
        return None

    # 6. Title checks
    current_title = str(profile.get("current_title", "")).lower()
    disqualified_titles = {
        "marketing manager",
        "graphic designer",
        "accountant",
        "operations manager",
        "customer support",
    }
    if any(t in current_title for t in disqualified_titles):
        return None

    # Base scoring
    score = 0.0

    # Title match bonus
    match_titles = ["engineer", "developer", "scientist", "analyst"]
    if any(t in current_title for t in match_titles):
        score += 20.0
    if (
        "ai" in current_title
        or "ml" in current_title
        or "machine learning" in current_title
        or "nlp" in current_title
        or "search" in current_title
    ):
        score += 25.0

    # Experience scoring
    if 5.0 <= exp_years <= 9.0:
        score += 25.0
    else:
        score += 10.0

    # Skill matching
    required_skills = {
        "python",
        "embeddings",
        "sentence-transformers",
        "vector database",
        "qdrant",
        "pinecone",
        "milvus",
        "weaviate",
        "elasticsearch",
        "opensearch",
        "search",
        "information retrieval",
        "ndcg",
        "mrr",
        "map",
    }
    preferred_skills = {
        "llm",
        "fine-tuning",
        "lora",
        "qlora",
        "peft",
        "learning-to-rank",
        "ltr",
        "xgboost",
        "lightgbm",
    }

    matching_req = []
    matching_pref = []

    for s in skills:
        name = str(s.get("name", "")).lower()
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)

        is_req = any(req in name for req in required_skills)
        is_pref = any(pref in name for pref in preferred_skills)

        skill_pts = 0
        if is_req:
            skill_pts += 5.0
            matching_req.append(s.get("name"))
        elif is_pref:
            skill_pts += 3.0
            matching_pref.append(s.get("name"))

        if skill_pts > 0:
            if prof == "expert":
                skill_pts += 2.0
            elif prof == "advanced":
                skill_pts += 1.0
            skill_pts *= 1.0 + min(dur / 60.0, 1.0)

        score += skill_pts

    # Behavioral signal modifiers
    rr = signals.get("recruiter_response_rate", 0.0)
    score *= 0.5 + 0.5 * rr

    last_act = parse_date(signals.get("last_active_date"))
    if last_act:
        days_inactive = (datetime(2026, 7, 2) - last_act).days
        if days_inactive > 180:
            score *= 0.5
        elif days_inactive <= 30:
            score *= 1.2

    if signals.get("open_to_work_flag"):
        score *= 1.1

    np_days = signals.get("notice_period_days", 90)
    if np_days <= 30:
        score += 5.0
    elif np_days > 90:
        score *= 0.8

    loc = str(profile.get("location", "")).lower()
    if any(
        city in loc for city in ["pune", "noida", "delhi", "gurgaon", "ncr", "mumbai"]
    ):
        score += 5.0
    elif signals.get("willing_to_relocate"):
        score += 3.0

    # Build detailed reasoning (1-2 sentences)
    req_str = (
        ", ".join(matching_req[:3]) if matching_req else "matching core NLP skills"
    )
    pref_str = f" and {matching_pref[0]}" if matching_pref else ""
    reason = f"{profile.get('current_title')} with {exp_years} years experience; expert in {req_str}{pref_str}; located in {profile.get('location')} with notice period of {np_days} days."

    return {"candidate_id": cid, "score": score, "reasoning": reason}


def main():
    parser = argparse.ArgumentParser(
        description="NexusMatch Engine Ranker for Redrob Hackathon"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Path to write the submission CSV"
    )
    args = parser.parse_args()

    scored = []

    # Handle both plain and gzipped files
    is_gzip = args.candidates.endswith(".gz")
    open_func = gzip.open if is_gzip else open
    mode = "rt" if is_gzip else "r"

    with open_func(args.candidates, mode, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            res = evaluate_candidate(cand)
            if res:
                scored.append(res)

    # Sort by score descending, then candidate_id ascending for deterministic tie-breaking
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Ensure parent directory exists for output
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write top 100
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, item in enumerate(scored[:100], 1):
            writer.writerow(
                [item["candidate_id"], idx, round(item["score"], 4), item["reasoning"]]
            )

    print(f"Successfully ranked candidates. Top 100 written to {args.out}")


if __name__ == "__main__":
    main()
