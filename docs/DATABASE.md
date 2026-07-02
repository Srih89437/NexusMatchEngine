# Relational Database Documentation

- **ORM Framework**: SQLAlchemy.
- **Connection pooling**: Leverages `pool_pre_ping=True` validation checks.
- **Failover**: Switches dynamically to SQLite memory when PostgreSQL is unavailable.
- **Schemas**:
  - `Candidate`: IDs, names, emails, raw text contents, experience histories.
  - `JobDescription`: IDs, titles, required skills list, min experience limits.
