# Docker Containers Deployment Reference

- **Containers Configured**: Exposes gateway backend API, recruiter UI dashboard, celery worker, Redis brokers, Qdrant vectors indexes, and PostgreSQL database.
- **Compose settings**: Checks postgres and redis check health statuses before executing background worker ingestion queues.
