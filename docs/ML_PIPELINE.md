# Machine Learning Model & Refinement

- **Model framework**: LightGBM LambdaMART ranker.
- **Objective**: `lambdarank` (NDCG@10 evaluated).
- **Features Matrix**: 14 normalized LTR feature metrics.
- **SHAP TreeExplainer**: Explains local candidate weights deviations.
- **Listwise LLM**: Reranks top candidates list and generates rationales.
- **Fact checking**: Validates rationales using deterministic span matching.
