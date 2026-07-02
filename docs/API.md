# API Reference Documentation

## 1. Authentication
- No active API authentication checks are configured in development.
- For production, configure OAuth2 / API keys.

---

## 2. API Endpoints

### GET `/health`
- **Description**: Returns system and connection states.
- **Response Example**:
  ```json
  {
    "status": "healthy",
    "details": {
      "database": "online",
      "qdrant": "online"
    }
  }
  ```

### POST `/api/v1/jobs/upload`
- **Description**: Persists job details.
- **Request Body**:
  ```json
  {
    "title": "Data Engineer",
    "required_skills": ["Python", "SQL", "Spark"],
    "min_experience_years": 3,
    "full_text": "Looking for Python SQL Spark engineer..."
  }
  ```
- **Response**:
  ```json
  {
    "job_id": "job_3ae847",
    "status": "stored"
  }
  ```
