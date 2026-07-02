import logging
import hashlib
from typing import Dict, List, Any, Optional
from src.config import settings

try:
    from FlagEmbedding import BGEM3FlagModel

    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGEM3Embedder:
    """BGE-M3 multi-vector embedding engine supporting Dense, Sparse, & Late-interaction tensors."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fp16: bool = False,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or "BAAI/bge-m3"
        self.use_fp16 = use_fp16
        self.device = device
        self.dimension = 1024

        if not FLAG_EMBEDDING_AVAILABLE:
            logger.warning(
                "FlagEmbedding library not available. Using deterministic mock embeddings."
            )
            self.model = None
            self.is_fallback = True
            return

        self.model_name = model_name or settings.BGE_MODEL_NAME
        self.use_fp16 = use_fp16
        self.device = device
        self.dimension = 1024

        logger.info(
            f"Loading native BGE-M3 model: {self.model_name} on device: {self.device or 'auto'}..."
        )
        try:
            self.model = BGEM3FlagModel(
                self.model_name, use_fp16=self.use_fp16, device=self.device
            )
            logger.info("BGE-M3 Model loaded natively successfully.")
            self.is_fallback = False
        except Exception as e:
            logger.warning(
                f"Could not load native BGE-M3 model ({e}). "
                "Falling back to a local deterministic mock embedding engine."
            )
            self.model = None
            self.is_fallback = True

    def generate_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate multi-vector embeddings for given text payloads.

        Returns:
            List of dicts containing dense vector representation,
            sparse tokens weights mapping, and late-interaction token tensors.
        """
        if self.is_fallback:
            import numpy as np

            results = []
            for text in texts:
                hashed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
                np.random.seed(hashed % (2**32 - 1))
                dense = np.random.uniform(-1.0, 1.0, self.dimension).tolist()
                indices = [101, 102, 103, 104, 105]
                values = [0.9, 0.8, 0.7, 0.6, 0.5]
                sparse = {"indices": indices, "values": values}
                colbert = [np.random.uniform(-1.0, 1.0, self.dimension).tolist()]
                results.append(
                    {"dense": dense, "sparse": sparse, "late_interaction": colbert}
                )
            logger.info(
                f"Generated mock deterministic BGE-M3 embeddings for {len(texts)} texts."
            )
            return results

        # Encode text using the native BGE-M3 model
        output = self.model.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=True
        )

        results = []
        for i in range(len(texts)):
            dense = output["dense_vecs"][i].tolist()

            # Convert lexical weights dict to Qdrant sparse format: {indices: [int], values: [float]}
            sparse_dict = output["lexical_weights"][i]
            indices = []
            values = []

            for token, weight in sparse_dict.items():
                try:
                    # If token is a string representation of ID, cast to int
                    token_id = int(token)
                    indices.append(token_id)
                    values.append(float(weight))
                except ValueError:
                    # Hash the token string to get a deterministic index within the BGE-M3 vocabulary limits
                    token_id = (
                        int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 30522
                    )
                    indices.append(token_id)
                    values.append(float(weight))

            # Sort by indices for Qdrant validation requirements
            if indices:
                sorted_pairs = sorted(zip(indices, values))
                indices, values = zip(*sorted_pairs)
                sparse = {"indices": list(indices), "values": list(values)}
            else:
                sparse = {"indices": [101], "values": [1.0]}

            # Parse late-interaction embeddings (ColBERT vectors)
            colbert = output["colbert_vecs"][i]
            if hasattr(colbert, "tolist"):
                colbert = colbert.tolist()

            results.append(
                {"dense": dense, "sparse": sparse, "late_interaction": colbert}
            )

        logger.info(f"Generated native BGE-M3 embeddings for {len(texts)} texts.")
        return results
