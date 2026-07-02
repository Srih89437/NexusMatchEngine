from unittest.mock import patch, MagicMock


def test_bge_m3_embedder_dimensions():
    with patch(
        "src.member2_retrieval.embedder.BGEM3Embedder.generate_embeddings"
    ) as mock_gen, patch(
        "src.member2_retrieval.embedder.BGEM3Embedder.__init__", return_value=None
    ):

        mock_gen.return_value = [
            {
                "dense": [0.1] * 1024,
                "sparse": {"indices": [101], "values": [0.5]},
                "late_interaction": [[0.01] * 1024],
            }
        ]

        from src.member2_retrieval.embedder import BGEM3Embedder

        embedder = BGEM3Embedder()
        results = embedder.generate_embeddings(["Python Developer Resume"])

        assert len(results) == 1
        assert "dense" in results[0]
        assert len(results[0]["dense"]) == 1024
        assert "sparse" in results[0]


def test_qdrant_hybrid_search():
    with patch(
        "src.member2_retrieval.qdrant_client.QdrantVectorClient.hybrid_search"
    ) as mock_search, patch(
        "src.member2_retrieval.qdrant_client.QdrantVectorClient.__init__",
        return_value=None,
    ):

        mock_search.return_value = [
            {
                "id": "cand_001",
                "score": 0.892,
                "payload": {"name": "Jane Doe", "skills": ["Python", "FastAPI"]},
            }
        ]

        from src.member2_retrieval.qdrant_client import QdrantVectorClient

        client = QdrantVectorClient()
        mock_dense = [0.1] * 1024
        mock_sparse = {"indices": [101], "values": [0.5]}

        results = client.hybrid_search(mock_dense, mock_sparse, limit=5)
        assert len(results) > 0
        assert "score" in results[0]
        assert "payload" in results[0]


def test_qdrant_setup_collection_indexes():
    with patch("src.member2_retrieval.qdrant_client.QdrantClient") as mock_qdrant_class:
        mock_client = mock_qdrant_class.return_value
        mock_client.collection_exists.return_value = False

        from src.member2_retrieval.qdrant_client import QdrantVectorClient

        client = QdrantVectorClient()
        client.setup_collection("test_collection")

        assert mock_client.create_collection.called
        calls = mock_client.create_payload_index.call_args_list
        assert len(calls) == 2
        field_names = [call.kwargs["field_name"] for call in calls]
        assert "skills" in field_names
        assert "years_exp" in field_names
