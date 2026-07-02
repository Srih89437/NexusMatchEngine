# Clean Architecture & System Design

The project uses a clean architecture structure dividing the pipeline logically:
- **Presentation**: FastAPI server and Streamlit dashboard.
- **Application**: Celery orchestrator pipelines.
- **Domain**: Pydantic candidate profiles schemas.
- **Infrastructure**: PostgreSQL, Qdrant client, and BGE-M3 models.
