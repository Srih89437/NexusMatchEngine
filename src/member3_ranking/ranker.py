import os
import logging
from typing import List, Dict, Any

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMRanker:
    """Real-time scoring service serving LambdaMART outputs and SHAP explanations."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "model", "lambdamart_model.txt"
        )
        self.gbm_model = None
        self.lightgbm_available = LIGHTGBM_AVAILABLE

        if self.lightgbm_available and os.path.exists(self.model_path):
            try:
                logger.info(
                    f"Loading native LightGBM LTR booster from {self.model_path}..."
                )
                self.gbm_model = lgb.Booster(model_file=self.model_path)
                logger.info("LightGBM LTR model loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Failed to load LightGBM model: {e}. Falling back to dynamic feature weights."
                )
        else:
            logger.info(
                "Using dynamic feature weights for matching calculations (LightGBM fallback)."
            )

    def predict_ranking(
        self, candidate_features: List[Dict[str, float]]
    ) -> List[float]:
        """Evaluate candidates based on calculated LTR feature metrics.

        Returns:
            List of float scores indicating match probabilities.
        """
        logger.info(
            f"Running inference for {len(candidate_features)} candidate feature matrix rows"
        )

        features_keys = [
            "experience_years",
            "skill_match_score",
            "trajectory_velocity",
            "exp_ratio",
            "skill_decay",
            "semantic_similarity",
            "dense_similarity",
            "sparse_similarity",
            "average_tenure",
            "job_transition_rate",
            "skill_recency",
            "education_match",
            "location_match",
            "hard_constraint_match",
        ]

        if self.gbm_model is not None:
            try:
                data = [
                    [row.get(k, 0.0) for k in features_keys]
                    for row in candidate_features
                ]
                predictions = self.gbm_model.predict(data)
                return [float(p) for p in predictions]
            except Exception as e:
                logger.error(
                    f"LightGBM prediction failed: {e}. Falling back to dynamic scoring."
                )

        # Dynamic linear LTR combination fallback
        weights = {
            "skill_match_score": 0.25,
            "semantic_similarity": 0.20,
            "dense_similarity": 0.15,
            "sparse_similarity": 0.10,
            "trajectory_velocity": 0.05,
            "exp_ratio": 0.05,
            "skill_decay": 0.03,
            "average_tenure": 0.03,
            "job_transition_rate": 0.02,
            "skill_recency": 0.02,
            "education_match": 0.03,
            "location_match": 0.02,
            "hard_constraint_match": 0.05,
        }

        scores = []
        for feat in candidate_features:
            score = 0.0
            for k, w in weights.items():
                score += feat.get(k, 0.0) * w
            scores.append(float(score))
        return scores

    def generate_shap_values(self, features: Dict[str, float]) -> Dict[str, float]:
        """Generate explainability panel contributions (SHAP) for a scoring round.

        Calculates impact dynamically based on deviation from standard baseline or via SHAP booster package.
        """
        logger.info("Computing SHAP force feature attributions...")

        features_keys = [
            "experience_years",
            "skill_match_score",
            "trajectory_velocity",
            "exp_ratio",
            "skill_decay",
            "semantic_similarity",
            "dense_similarity",
            "sparse_similarity",
            "average_tenure",
            "job_transition_rate",
            "skill_recency",
            "education_match",
            "location_match",
            "hard_constraint_match",
        ]

        if self.gbm_model is not None:
            try:
                import shap
                import numpy as np

                explainer = shap.TreeExplainer(self.gbm_model)
                row = [features.get(k, 0.0) for k in features_keys]
                shap_values = explainer.shap_values(np.array([row]))
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                return {
                    k: float(shap_values[0][idx]) for idx, k in enumerate(features_keys)
                }
            except Exception as e:
                logger.warning(
                    f"Native SHAP computation failed: {e}. Falling back to dynamic baseline attribution."
                )

        # Baselines representing typical/average candidate parameters
        baseline = {
            "skill_match_score": 0.50,
            "semantic_similarity": 0.50,
            "trajectory_velocity": 1.00,
            "exp_ratio": 1.00,
            "skill_decay": 0.90,
            "dense_similarity": 0.50,
            "sparse_similarity": 0.50,
            "average_tenure": 0.50,
            "job_transition_rate": 0.50,
            "skill_recency": 0.90,
            "education_match": 0.50,
            "location_match": 0.50,
            "hard_constraint_match": 0.50,
        }

        # Positive/negative impact coefficient per feature unit deviation
        weights = {
            "skill_match_score": 0.40,
            "semantic_similarity": 0.30,
            "trajectory_velocity": 0.15,
            "exp_ratio": 0.10,
            "skill_decay": -0.05,
            "dense_similarity": 0.20,
            "sparse_similarity": 0.10,
            "average_tenure": 0.15,
            "job_transition_rate": -0.10,
            "skill_recency": 0.05,
            "education_match": 0.10,
            "location_match": 0.05,
            "hard_constraint_match": 0.20,
        }

        shap_values = {}
        for k, w in weights.items():
            val = features.get(k, 0.0)
            base_val = baseline.get(k, 0.0)
            shap_values[k] = float((val - base_val) * w)

        return shap_values
