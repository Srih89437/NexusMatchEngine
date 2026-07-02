# Hybrid Vector Retrieval Design

- **Vector Database**: Qdrant.
- **Embedder**: BGE-M3 model generating 1024-dimensional dense vectors and named sparse lexical weights.
- **RRF (Reciprocal Rank Fusion)**: Combines dense query outputs and sparse query outputs.
- **Indexes**: Keywords indexes on `skills` and Float indexes on `years_exp` payload keys on collection setup.
