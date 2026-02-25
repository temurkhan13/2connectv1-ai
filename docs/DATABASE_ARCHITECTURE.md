# Reciprocity Platform - Database Architecture

## CRITICAL: Two Separate PostgreSQL Databases

The platform uses **TWO SEPARATE PostgreSQL databases**. Confusing them causes production bugs.

```
+---------------------------+       +---------------------------+
|   BACKEND DATABASE        |       |   AI DATABASE             |
|   (reciprocity_db)        |       |   (reciprocity_ai)        |
+---------------------------+       +---------------------------+
|   Port: 5432              |       |   Port: 5433              |
|   User: postgres          |       |   User: reciprocity_user  |
|   Pass: postgres          |       |   Pass: reciprocity_pass  |
+---------------------------+       +---------------------------+
|                           |       |                           |
|   users                   |       |   embeddings              |
|   user_summaries          |       |   pgvector indexes        |
|   matches                 |       |   (vectors only)          |
|   messages                |       |                           |
|   profiles                |       |                           |
|   (all business data)     |       |                           |
|                           |       |                           |
+---------------------------+       +---------------------------+
```

## Environment Variables

```bash
# AI Database (port 5433) - embeddings and vectors
DATABASE_URL=postgresql://reciprocity_user:reciprocity_pass@localhost:5433/reciprocity_ai

# Backend Database (port 5432) - users, matches, profiles
RECIPROCITY_BACKEND_DB_URL=postgresql://postgres:postgres@host.docker.internal:5432/reciprocity_db
```

## When to Use Which Database

### Use AI Database (`DATABASE_URL` / `get_connection()`)
- Storing embeddings (pgvector)
- Vector similarity searches
- AI-specific data that doesn't need to be in backend

### Use Backend Database (`RECIPROCITY_BACKEND_DB_URL` / `get_backend_connection()`)
- Reading/writing user records (`users` table)
- Reading/writing user summaries (`user_summaries` table)
- Reading/writing matches (`matches` table)
- Any data that the NestJS backend also uses

## Code Pattern: postgresql.py

The `PostgreSQLAdapter` class provides two connection methods:

```python
class PostgreSQLAdapter:
    def get_connection(self):
        """AI database (port 5433) - for embeddings"""
        return psycopg2.connect(self.database_url)

    def get_backend_connection(self):
        """Backend database (port 5432) - for users/matches"""
        return psycopg2.connect(self.backend_database_url)
```

### Methods That MUST Use Backend Connection

| Method | Table | Why |
|--------|-------|-----|
| `update_user_onboarding_status()` | `users` | Users table is in backend DB |
| `create_user_summary()` | `user_summaries` | Summaries displayed by backend |
| `get_valid_user_ids()` | `users` | Validates users for matching |

## Common Mistakes (P0 Bugs)

### BUG: Using `get_connection()` for user operations

```python
# WRONG - This will fail silently or throw "relation does not exist"
conn = self.get_connection()  # AI DB
cursor.execute("UPDATE users SET onboarding_status = 'completed'")  # FAILS

# CORRECT - Use backend connection
conn = self.get_backend_connection()  # Backend DB
cursor.execute("UPDATE users SET onboarding_status = 'completed'")  # Works
```

**Symptoms of this bug:**
- Onboarding appears to complete but user status doesn't change
- User gets stuck in "in_progress" state
- Redirect loops after onboarding
- "relation 'users' does not exist" errors in logs

### BUG: Missing `RECIPROCITY_BACKEND_DB_URL` in Docker

```yaml
# docker-compose.yml - ai-service environment
environment:
  DATABASE_URL: postgresql://reciprocity_user:reciprocity_pass@db:5432/reciprocity_ai
  # MISSING: RECIPROCITY_BACKEND_DB_URL  <-- WILL BREAK USER OPERATIONS
```

**Fix:**
```yaml
environment:
  DATABASE_URL: postgresql://reciprocity_user:reciprocity_pass@db:5432/reciprocity_ai
  RECIPROCITY_BACKEND_DB_URL: postgresql://postgres:postgres@host.docker.internal:5432/reciprocity_db
```

## Docker Networking

From inside Docker containers:
- `db:5432` = AI database container (reciprocity_ai)
- `host.docker.internal:5432` = Host machine's backend database (reciprocity_db)

```
+------------------+       +------------------+       +------------------+
|  ai-service      |       |  db container    |       |  host machine    |
|  (Docker)        |       |  (Docker)        |       |  (Windows/Mac)   |
+------------------+       +------------------+       +------------------+
        |                          |                          |
        |   db:5432 (AI DB)        |                          |
        +------------------------->+                          |
        |                                                     |
        |   host.docker.internal:5432 (Backend DB)            |
        +---------------------------------------------------->+
```

## Testing Database Connections

```bash
# Test AI database (from host)
psql -h localhost -p 5433 -U reciprocity_user -d reciprocity_ai

# Test backend database (from host)
psql -h localhost -p 5432 -U postgres -d reciprocity_db

# Test from inside Docker container
docker exec reciprocity-ai-service python -c "
from app.adapters.postgresql import postgresql_adapter
# This should work
ai_conn = postgresql_adapter.get_connection()
ai_conn.close()
print('AI DB: OK')

# This should also work
backend_conn = postgresql_adapter.get_backend_connection()
backend_conn.close()
print('Backend DB: OK')
"
```

## Incident Log

| Date | Bug | Root Cause | Fix |
|------|-----|------------|-----|
| 2026-02-17 | Onboarding completion reset loop (P0 SHOWSTOPPER) | `update_user_onboarding_status()` used `get_connection()` (AI DB) instead of `get_backend_connection()` (Backend DB) | Added `get_backend_connection()` method, updated method to use it |
| 2026-02-17 | Match sync failing | Missing `RECIPROCITY_BACKEND_DB_URL` in docker-compose.yml | Added env var to ai-service |

## Checklist for New Features

When writing code that touches user data:

- [ ] Am I using the right database connection?
- [ ] Is `RECIPROCITY_BACKEND_DB_URL` set in docker-compose.yml?
- [ ] Is `RECIPROCITY_BACKEND_DB_URL` set in .env?
- [ ] Am I using `get_backend_connection()` for backend tables?
- [ ] Am I using `get_connection()` for AI-only data (embeddings)?

## Files to Update Together

When changing database configuration:

1. `app/adapters/postgresql.py` - Connection methods
2. `.env` - Local environment
3. `docker-compose.yml` - Docker environment (ai-service AND celery-worker)
4. `.env.production` - Production environment (if exists)
