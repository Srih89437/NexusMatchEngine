import logging
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import settings

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        SparseVectorParams,
        SparseIndexParams,
        PointStruct,
        SparseVector,
        NamedSparseVector,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_hits: List[Any],
    sparse_hits: List[Any],
    limit: int = 10,
    rrf_constant: int = 60,
) -> List[Dict[str, Any]]:
    """Perform Reciprocal Rank Fusion (RRF) on dense and sparse hits.

    Formula:
        RRF = 1 / (60 + Rank_d) + 1 / (60 + Rank_s)
    """
    scores = {}
    payloads = {}

    # Process dense rank hits
    for rank, hit in enumerate(dense_hits):
        point_id = hit.id
        scores[point_id] = scores.get(point_id, 0.0) + (1.0 / (rrf_constant + rank + 1))
        payloads[point_id] = getattr(hit, "payload", {})

    # Process sparse rank hits
    for rank, hit in enumerate(sparse_hits):
        point_id = hit.id
        scores[point_id] = scores.get(point_id, 0.0) + (1.0 / (rrf_constant + rank + 1))
        payloads[point_id] = getattr(hit, "payload", {})

    # Sort and slice
    sorted_points = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    return [
        {"id": point_id, "score": score, "payload": payloads[point_id]}
        for point_id, score in sorted_points
    ]


_client_cache = {}


class QdrantVectorClient:
    """Client handling hybrid vector collection setup, indexes, and queries in Qdrant."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "The 'qdrant-client' library is required to run QdrantVectorClient. Please install it using pip."
            )

        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT

        logger.info(f"Connecting to Qdrant cluster at: {self.host}:{self.port}...")
        try:
            cache_key = f"server:{self.host}:{self.port}"
            if cache_key in _client_cache:
                self.client = _client_cache[cache_key]
            else:
                self.client = QdrantClient(host=self.host, port=self.port, timeout=5.0)
                # Check connection integrity
                self.client.get_collections()
                _client_cache[cache_key] = self.client
            logger.info("Successfully connected to Qdrant cluster.")
        except Exception as e:
            logger.warning(
                f"Could not connect to Qdrant server at {self.host}:{self.port}: {e}. "
                "Falling back to an in-memory Qdrant instance."
            )
            cache_key = "local:data/qdrant_local"
            if cache_key in _client_cache:
                self.client = _client_cache[cache_key]
            else:
                self.client = QdrantClient(path="data/qdrant_local")
                _client_cache[cache_key] = self.client
            _client_cache[f"server:{self.host}:{self.port}"] = self.client
            try:
                self.setup_collection()
            except Exception as setup_err:
                logger.error(
                    f"Failed to auto-setup collection on in-memory Qdrant instance: {setup_err}"
                )

    def setup_collection(
        self, collection_name: str = settings.QDRANT_COLLECTION
    ) -> None:
        """Create hybrid dense/sparse collection configuration in Qdrant if non-existent."""
        try:
            if self.client.collection_exists(collection_name):
                logger.info(
                    f"Collection '{collection_name}' already exists. Skipping recreation."
                )
                return

            logger.info(
                f"Initializing collection '{collection_name}' in Qdrant (Dense: 1024-dim, Sparse enabled)..."
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                },
            )

            # Create payload indexes for fast filtering
            from qdrant_client.models import PayloadSchemaType

            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="skills",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="years_exp",
                field_schema=PayloadSchemaType.FLOAT,
            )

            logger.info(
                f"Collection '{collection_name}' and payload indexes created/reset successfully."
            )
        except Exception as e:
            logger.error(f"Failed to setup Qdrant collection '{collection_name}': {e}")
            raise

    def upsert_vectors(
        self,
        points: List[Dict[str, Any]],
        collection_name: str = settings.QDRANT_COLLECTION,
    ) -> bool:
        """Upload payloads containing dense and sparse vectors to vector search index.

        Each point in `points` dict should look like:
        {
            "id": str,
            "dense": List[float],
            "sparse": {"indices": List[int], "values": List[float]},
            "payload": Dict[str, Any]
        }
        """
        if not points:
            return True

        try:
            import uuid as _uuid

            def _to_uuid(id_str: str) -> str:
                """Convert arbitrary string ID to a deterministic UUID v5."""
                try:
                    _uuid.UUID(str(id_str))
                    return str(id_str)
                except ValueError:
                    return str(_uuid.uuid5(_uuid.NAMESPACE_DNS, str(id_str)))

            qdrant_points = [
                PointStruct(
                    id=_to_uuid(pt["id"]),
                    vector={
                        "dense": pt["dense"],
                        "sparse": SparseVector(
                            indices=pt["sparse"]["indices"],
                            values=pt["sparse"]["values"],
                        ),
                    },
                    payload=pt["payload"],
                )
                for pt in points
            ]
            self.client.upsert(collection_name=collection_name, points=qdrant_points)
            logger.info(
                f"Upserted {len(points)} vectors to collection '{collection_name}'"
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")
            return False

    def hybrid_search(
        self,
        dense_query: List[float],
        sparse_query: Dict[str, Any],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = settings.QDRANT_COLLECTION,
    ) -> List[Dict[str, Any]]:
        """Perform Approximate Nearest Neighbor (ANN) search merging dense and sparse scores.

        Returns:
            List of matching records and search similarity scores.
        """
        try:
            qdrant_filter = None
            if filters:
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
                qdrant_filter = Filter(must=conditions)

            # 1. Query Dense space
            if hasattr(self.client, "search"):
                dense_hits = self.client.search(
                    collection_name=collection_name,
                    query_vector=("dense", dense_query),
                    limit=limit * 2,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
            else:
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=dense_query,
                    using="dense",
                    limit=limit * 2,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                dense_hits = response.points

            # 2. Query Sparse space
            if hasattr(self.client, "search"):
                sparse_hits = self.client.search(
                    collection_name=collection_name,
                    query_vector=NamedSparseVector(
                        name="sparse",
                        vector=SparseVector(
                            indices=sparse_query["indices"],
                            values=sparse_query["values"],
                        ),
                    ),
                    limit=limit * 2,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
            else:
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=SparseVector(
                        indices=sparse_query["indices"], values=sparse_query["values"]
                    ),
                    using="sparse",
                    limit=limit * 2,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                sparse_hits = response.points

            # 3. Fuse scores via Reciprocal Rank Fusion (RRF)
            fused_results = reciprocal_rank_fusion(dense_hits, sparse_hits, limit)
            return fused_results
        except Exception as e:
            logger.error(f"Qdrant hybrid search failed: {e}")
            raise
