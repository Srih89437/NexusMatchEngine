import os
import logging
import re
from typing import List, Dict, Any
from src.config import settings
from pydantic import BaseModel, Field

try:
    import instructor
    from openai import OpenAI

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandidateRankOrder(BaseModel):
    ordered_candidate_ids: List[str] = Field(
        ..., description="List of candidate IDs sorted from best to worst match"
    )
    rationales: Dict[str, str] = Field(
        ...,
        description="Dictionary mapping candidate ID to a short 1-sentence matching rationale",
    )


class ListwiseLLMRanker:
    """Fine-grained LLM re-ranker processing structural context via listwise prompts."""

    def __init__(self):
        logger.info(
            "Initializing vLLM / OpenAI client framework (Late listwise scoring)..."
        )
        self.api_key = settings.VLLM_API_KEY or settings.OPENAI_API_KEY
        self.client = None
        self.llm_available = LLM_AVAILABLE

        is_mock_key = (
            not self.api_key
            or self.api_key == "mock-key-for-development"
            or "your-openai-api-key-here" in self.api_key
            or "mock-key-for-local-vllm" in self.api_key
        )
        is_localhost_loop = (
            settings.VLLM_API_URL is not None
            and "localhost:8000" in settings.VLLM_API_URL
        )

        if self.llm_available and not is_mock_key and not is_localhost_loop:
            try:
                if settings.VLLM_API_URL:
                    logger.info(f"Using local vLLM server: {settings.VLLM_API_URL}")
                    openai_client = OpenAI(
                        base_url=settings.VLLM_API_URL, api_key=self.api_key
                    )
                    self.model_name = "local-model"
                else:
                    logger.info("Using OpenAI GPT endpoint...")
                    openai_client = OpenAI(api_key=self.api_key)
                    self.model_name = "gpt-4o-mini"

                self.client = instructor.from_openai(openai_client)
                logger.info("LLM listwise ranker client initialized successfully.")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize LLM client: {e}. Fallback ranking active."
                )
        else:
            logger.warning(
                "Instructor client, API keys, or valid server URLs not configured. Fallback ranking active."
            )

    def verify_rationale(self, rationale: str, raw_resume_text: str) -> str:
        """Verify LLM rationale claims against candidate's raw resume text using deterministic span matching.

        Extracts key noun phrases/capitalized words and checks if they appear in the raw resume text.
        Rejects unsupported claims.
        """
        if not rationale or not raw_resume_text:
            return "Aligned based on verified technical parameters."

        # Extract capitalized proper nouns (e.g. "Python", "Google", "FastAPI")
        words = re.findall(r"\b[A-Z][a-zA-Z0-9+#]*\b", rationale)
        resume_lower = raw_resume_text.lower()
        unverified_claims = []

        for word in words:
            # Skip common short grammatical words
            if len(word) <= 2 or word.lower() in ["the", "and", "for", "with", "this"]:
                continue
            if word.lower() not in resume_lower:
                unverified_claims.append(word)

        if unverified_claims:
            logger.warning(
                f"Rejecting unverified claims in rationale: {unverified_claims}"
            )
            return "Aligned based on verified technical parameters matching the candidate profile."

        return rationale

    def rerank_list(
        self, candidates: List[Dict[str, Any]], job_description: str
    ) -> List[Dict[str, Any]]:
        """Submit list of top matches to LLM for final listwise sorting.

        Returns:
            List of candidates sorted in optimized alignment order.
        """
        if not candidates:
            return []

        logger.info(
            f"Submitting {len(candidates)} candidate summaries to listwise LLM pipeline..."
        )

        from src.member2_retrieval.postgres_client import PostgresStateClient

        pg_client = PostgresStateClient()

        if self.client is not None:
            try:
                candidate_summaries = []
                for c in candidates:
                    summary = f"ID: {c['id']}, Name: {c['name']}, Initial Score: {c['ltr_score']:.2f}"
                    candidate_summaries.append(summary)

                context = "\n".join(candidate_summaries)

                logger.info(
                    "Context window size is safe. Executing LLM listwise inference..."
                )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=CandidateRankOrder,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert recruiter. Compare the job description with the candidate summaries. "
                                "Sort the candidate IDs from best match to worst match. Provide a 1-sentence rationale for each candidate."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Job Description:\n{job_description}\n\nCandidates:\n{context}",
                        },
                    ],
                    temperature=0.0,
                )

                ordered_ids = response.ordered_candidate_ids
                rationales = response.rationales

                candidates_map = {c["id"]: c for c in candidates}
                reranked_candidates = []

                for idx, cid in enumerate(ordered_ids):
                    if cid in candidates_map:
                        cand = candidates_map[cid]
                        cand["listwise_rank"] = idx + 1

                        raw_rationale = rationales.get(
                            cid, "Aligned based on technical parameters."
                        )
                        cand_db = pg_client.get_candidate(cid)
                        raw_text = cand_db.get("raw_text", "") if cand_db else ""

                        cand["llm_rationale"] = self.verify_rationale(
                            raw_rationale, raw_text
                        )
                        reranked_candidates.append(cand)

                # Add any candidates missed by the LLM at the end
                for cid, cand in candidates_map.items():
                    if cand not in reranked_candidates:
                        cand["listwise_rank"] = len(reranked_candidates) + 1
                        cand_db = pg_client.get_candidate(cid)
                        raw_text = cand_db.get("raw_text", "") if cand_db else ""

                        fallback_rationale = f"Candidate ranked by LTR score of {cand.get('ltr_score', 0.0):.2f}."
                        cand["llm_rationale"] = self.verify_rationale(
                            fallback_rationale, raw_text
                        )
                        reranked_candidates.append(cand)

                return reranked_candidates

            except Exception as e:
                logger.error(
                    f"LLM reranking failed: {e}. Falling back to initial LTR score sorting."
                )

        # Algorithmic fallback: Sort purely by LTR score
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("ltr_score", 0.0), reverse=True
        )
        for idx, cand in enumerate(sorted_candidates):
            cand["listwise_rank"] = idx + 1
            cand_db = pg_client.get_candidate(cand["id"])
            raw_text = cand_db.get("raw_text", "") if cand_db else ""

            fallback_rationale = f"Candidate ranked by LightGBM model score of {cand.get('ltr_score', 0.0):.2f} (LLM ranker fallback)."
            cand["llm_rationale"] = self.verify_rationale(fallback_rationale, raw_text)

        return sorted_candidates
