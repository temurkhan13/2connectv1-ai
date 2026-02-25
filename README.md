# Reciprocity AI

AI-powered reciprocity platform for persona generation and matching.

> **CRITICAL: Read Before Writing Database Code**
>
> This platform uses **TWO SEPARATE PostgreSQL databases**. Using the wrong one causes P0 bugs.
>
> | Database | Port | Use For |
> |----------|------|---------|
> | Backend (`reciprocity_db`) | 5432 | Users, matches, profiles |
> | AI (`reciprocity_ai`) | 5433 | Embeddings only |
>
> **See [docs/DATABASE_ARCHITECTURE.md](docs/DATABASE_ARCHITECTURE.md) before touching database code.**

## Prerequisites

### Development Environment (Local Only)
- **Python**: 3.12+ (tested with Python 3.12.3)
- **pip**: 24.0+ (for installing uv)
- **uv**: 0.8+ (tested with uv 0.8.15)
- **Docker and Docker Compose**: For local development services only (PostgreSQL, Redis, LocalStack)
- **AWS CLI**: Configured with test credentials for LocalStack

### Production Environment (No Docker)
- **Python**: 3.12+ 
- **pip**: 24.0+ (for installing uv)
- **uv**: 0.8+ (fast package manager)
- **PostgreSQL**: 12+ with pgvector extension (managed service or dedicated server)
- **Redis**: 6.0+ (managed service or dedicated server)
- **AWS**: EC2 instance with IAM role for DynamoDB access

**Important:** Docker is used **only for local development** to run PostgreSQL, Redis, and LocalStack. Production deployments use managed services or dedicated servers instead of Docker containers.

### Version Check

Verify your environment meets the requirements:

```bash
# Check Python version
python3 --version  # Should show Python 3.12+

# Check pip version  
pip --version      # Should show pip 24.0+

# Install and check uv version
pip install uv
uv --version       # Should show uv 0.8+

# Check Docker (for local development only)
docker --version && docker compose version
```

## AWS CLI Setup

Configure AWS CLI with test credentials for LocalStack:

```bash
aws configure
# AWS Access Key ID: test
# AWS Secret Access Key: test
# Default region name: us-east-1
# Default output format: json
```

## Important: Development vs Production

**This workflow is for LOCAL DEVELOPMENT ONLY:**
- Uses LocalStack to emulate DynamoDB and S3
- Creates tables via console commands
- Uses test AWS credentials
- **Data persists across LocalStack restarts** (configured with `PERSISTENCE=1`)

**Production:**
- Uses real AWS DynamoDB and S3
- Tables created via infrastructure (Terraform/CloudFormation)
- Real AWS credentials and regions
- See [Production Deployment](#production-deployment) section below

## Quick Start

### 1. Create DynamoDB Table
**IMPORTANT:** You must manually create the table using AWS CLI commands (see [Table Creation](#table-creation) section below). The `dev.sh` script only checks if the table exists.

### 2. Start Development Environment
```bash
./dev.sh
```

This will:
- Start Redis, PostgreSQL, and LocalStack (DynamoDB + S3 emulator) with **data persistence**
- Check if the table exists
- Start FastAPI server and Celery worker

**OR manually:**
```bash
# Start all services (Redis, PostgreSQL, LocalStack)
docker compose up -d

# Start server
uv run python scripts/start_server.py

# Start Celery worker (in another terminal)
uv run python scripts/start_celery.py
```

### 3. Test the API

#### Basic Test (without resume)
```bash
# Use your configured PORT (default: 8000)
curl -X POST "http://localhost:${PORT:-8000}/api/v1/user/register" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user-123",
    "questions": [
      {
        "prompt": "What is your experience level?",
        "answer": "5 years"
      },
      {
        "prompt": "What are you looking for?",
        "answer": "I need funding and mentorship for my startup"
      },
      {
        "prompt": "What can you offer?",
        "answer": "Technical expertise and product development skills"
      }
    ]
  }'
```

#### Test with Resume
```bash
# Use your configured PORT (default: 8000)
curl -X POST "http://localhost:${PORT:-8000}/api/v1/user/register" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user-with-resume-123",
    "resume_link": "https://example.com/resume.pdf",
    "questions": [
      {
        "prompt": "What is your experience level?",
        "answer": "5 years"
      }
    ]
  }'
```

#### Test with LocalStack S3 URL (after uploading resume)
```bash
# Use your configured PORT (default: 8000)
curl -X POST "http://localhost:${PORT:-8000}/api/v1/user/register" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-s3-user-123",
    "resume_link": "http://localhost:4566/reciprocity-resumes/resumes/your-resume.pdf",
    "questions": [
      {
        "prompt": "What is your experience level?",
        "answer": "5 years"
      }
    ]
  }'
```

**Note:** Replace `your-resume.pdf` with the actual filename of your uploaded resume.


### 4. Stop everything
```bash
# Stop server (Ctrl+C)
# Stop Celery worker (Ctrl+C)
# Stop all Docker services (Redis, PostgreSQL, LocalStack)
docker compose down
```

## AWS CLI Commands

### Create DynamoDB Tables

#### User Profiles Table
```bash
aws dynamodb create-table \
  --table-name user_profiles \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```

#### User Matches Table
```bash
aws dynamodb create-table \
  --table-name user_matches \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```

### Create S3 Bucket for Resume Testing
```bash
# Create S3 bucket for resume testing
aws --endpoint-url=http://localhost:4566 s3 mb s3://reciprocity-resumes

# Upload a test resume (replace with your resume file path)
aws --endpoint-url=http://localhost:4566 s3 cp "/path/to/your/resume.pdf" s3://reciprocity-resumes/resumes/

# List files in bucket
aws --endpoint-url=http://localhost:4566 s3 ls s3://reciprocity-resumes/resumes/
```

**Note:** The S3 bucket name `reciprocity-resumes` is fixed and used in the test files. Use this exact name for consistency.

### Verify Table was Created
```bash
aws dynamodb list-tables \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```

### Check Table Details
```bash
aws dynamodb describe-table \
  --table-name user_profiles \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```

### Delete Table (if needed)
```bash
aws dynamodb delete-table \
  --table-name user_profiles \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```

### Check LocalStack Health
```bash
curl http://localhost:4566/_localstack/health
```

## Table Creation

**Required:** You must create both DynamoDB tables manually using AWS CLI commands below.

### Create DynamoDB Tables

#### User Profiles Table
```bash
aws dynamodb create-table \
  --table-name user_profiles \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```

#### User Matches Table
```bash
aws dynamodb create-table \
  --table-name user_matches \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```

#### User Feedback Table
```bash
aws dynamodb create-table \
  --table-name user_feedback \
  --attribute-definitions AttributeName=feedback_id,AttributeType=S \
  --key-schema AttributeName=feedback_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```

#### AI Chat Records Table
```bash
aws dynamodb create-table \
  --table-name ai_chat_records \
  --attribute-definitions AttributeName=chat_id,AttributeType=S \
  --key-schema AttributeName=chat_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```

#### Notified Match Pairs Table
```bash
aws dynamodb create-table \
  --table-name notified_match_pairs \
  --attribute-definitions AttributeName=pair_key,AttributeType=S \
  --key-schema AttributeName=pair_key,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --endpoint-url http://localhost:4566 \
  --region us-east-1
```


## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

The `.env` file contains:
- **Application**: `APP_NAME=reciprocity-platform`, `APP_VERSION=1.0.0`, `ENVIRONMENT=development`, `LOG_LEVEL=INFO`, `PORT=8000`, `HOST=0.0.0.0`
- **CORS**: `CORS_ORIGINS=["http://localhost:3000"]`, `ALLOWED_HOSTS=["*"]`
- **Database**: `REDIS_URL=redis://localhost:6380/0` (Docker Redis on port 6380), `DYNAMO_PROFILE_TABLE_NAME=user_profiles`
- **PostgreSQL**: `DATABASE_URL=postgresql://reciprocity_user:reciprocity_pass@localhost:5433/reciprocity_ai`
- **Embeddings**: `EMBEDDING_MODEL=all-MiniLM-L6-v2`, `SIMILARITY_THRESHOLD=0.3`
- **AWS**: `AWS_ACCESS_KEY_ID=test`, `AWS_SECRET_ACCESS_KEY=test`, `AWS_DEFAULT_REGION=us-east-1`
- **LocalStack**: `DYNAMODB_ENDPOINT_URL=http://localhost:4566`
- **OpenAI**: `OPENAI_API_KEY=sk-proj-your-key`, `OPENAI_MODEL=gpt-4o-mini`
- **Backend Integration**: `RECIPROCITY_BACKEND_URL=http://localhost:3000/api` (optional)

**All hardcoded values have been replaced with environment variables:**
- Server port and host are now configurable via `PORT` and `HOST`
- Application name and version use `APP_NAME` and `APP_VERSION`
- CORS origins and allowed hosts are configurable via JSON arrays
- DynamoDB table name uses `DYNAMO_PROFILE_TABLE_NAME`
- Redis URL is configurable via `REDIS_URL`

## API Endpoints

### Core Endpoints
- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /docs` - API documentation

### User Management
- `POST /api/v1/user/register` - Register user with resume and questions
- `GET /api/v1/user/{user_id}` - Get user profile by ID
- `POST /api/v1/user/approve-summary` - Trigger embeddings generation and matching after persona approval
- `POST /api/v1/user/feedback` - Submit feedback on matches/chats to improve future matching

### Matchmaking & Embeddings
- `GET /api/v1/matching/{user_id}/matches` - Get all matches for a user (requirements + offerings)
- `GET /api/v1/matching/{user_id}/requirements-matches` - Get matches for user's requirements
- `GET /api/v1/matching/{user_id}/offerings-matches` - Get matches for user's offerings  
- `GET /api/v1/matching/stats` - Get matching system statistics

## Matchmaking & Embeddings System

The platform includes a sophisticated AI-powered matchmaking system that connects users based on their requirements and offerings using semantic similarity.

### Architecture Overview

The matchmaking system consists of several key components:

1. **Embedding Service** (`app/services/embedding_service.py`)
   - Uses SentenceTransformers for high-quality semantic embeddings
   - Model: `all-MiniLM-L6-v2` (384 dimensions, CPU-optimized)
   - Stores embeddings in PostgreSQL with pgvector extension
   - In-memory caching for performance

2. **Matching Router** (`app/routers/matching.py`)
   - RESTful API endpoints for retrieving matches
   - Supports configurable similarity thresholds and result limits
   - Returns matches with explanations and similarity scores

3. **Match Explanation Service** (`app/services/match_explanation_service.py`)
   - Generates human-readable explanations for matches
   - Quality assessment based on similarity scores
   - Match type descriptions (requirements-to-offerings, etc.)

4. **Filtering Service** (`app/services/filtering_service.py`)
   - Advanced filtering capabilities for persona matching
   - Multiple filter operators (equals, contains, range, regex, etc.)
   - Weighted scoring and pagination support

5. **Background Processing** (`app/workers/embedding_processing.py`)
   - Celery tasks for asynchronous embedding generation
   - Triggered after persona generation completes
   - Processes both user requirements and offerings

### How Matching Works

#### New User Registration Flow (`update: false`)

1. **User Registration**: User submits resume (optional) and answers questions
2. **Task Chain Execution**: Celery automatically chains three tasks:
   - Resume processing (parse and extract text, or skip if no resume)
   - Persona generation (AI creates persona with requirements/offerings)
   - Backend receives **persona ready notification**
3. **Manual Approval**: Backend calls `/api/v1/user/approve-summary` endpoint
4. **Embedding & Matching**: 
   - Generate semantic embeddings for requirements and offerings
   - Find matches against all existing users
   - Store matches in DynamoDB
   - Update reciprocal matches (add this user to matched users' stored matches)
5. **Storage**: Embeddings stored in PostgreSQL with pgvector indexing
6. **Explanation**: AI generates human-readable match explanations

#### Profile Update Flow (`update: true`)

1. **Profile Update**: User updates their profile via `/api/v1/user/register` with `update: true`
2. **Flag Set**: User's `needs_matchmaking` flag is set to `'true'` in DynamoDB
3. **Scheduled Worker** (3-Phase Sequential Processing):
   
   **PHASE 1: Resume + Persona** (parallel async chains)
   - Re-process resume for all users
   - Re-generate persona (no notification sent)
   - All chains complete before Phase 2
   
   **PHASE 2: Embeddings + Matching** (parallel, all embeddings ready)
   - Generate new embeddings for all users
   - Find new matches against ALL users (guaranteed all embeddings exist)
   - Store matches WITHOUT reciprocal updates yet
   - Updated users CAN match with each other reliably
   
   **PHASE 3: Reciprocal Updates + Notification** (sequential)
   - Update reciprocal matches for all processed users
   - Collect all match pairs
   - Send single batch notification to backend
   - Set `needs_matchmaking='false'` for all processed users
   
4. **Batch Notification**: Backend receives single notification at `/api/v1/webhooks/matches-ready` with all match pairs

#### Reciprocal Matching System

The system implements **reciprocal matching** to ensure all users stay up-to-date:

- **New User Joins**: 
  - New user gets matches with existing users
  - Existing users automatically receive this new user in their stored matches
  
- **User Updates Profile**:
  - Updated user gets fresh matches with all users
  - Users who match with the updated profile get them added to their stored matches

- **Old Unchanged Users**:
  - Do NOT trigger re-processing for themselves
  - Automatically receive new matches when:
    - New users join the platform
    - Existing users update their profiles
  - Their stored matches stay current without re-processing

**Notification Flow**:
- New users: Backend receives individual notifications via `/api/v1/webhooks/user-matches-ready`
- Scheduled updates: Backend receives batch notifications via `/api/v1/webhooks/matches-ready` with format:
  ```json
  {
    "batch_id": "task_id_123",
    "match_pairs": [
      {
        "user_a_id": "updated_user_123",
        "user_a_designation": "Software Engineer",
        "user_b_id": "old_user_456", 
        "user_b_designation": "Product Manager"
      }
    ]
  }
  ```

### Matching Types

- **Requirements to Offerings**: What you need ↔ What others provide
- **Offerings to Requirements**: What you provide ↔ What others need
- **Bidirectional**: Complete compatibility analysis

### Configuration

Key environment variables for the matching system:

```bash
# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2        # SentenceTransformers model
SIMILARITY_THRESHOLD=0.3                # Minimum match threshold (0.0-1.0)

# Database Configuration  
DATABASE_URL=postgresql://reciprocity_user:reciprocity_pass@localhost:5433/reciprocity_ai
```

### Example API Usage

#### Get All Matches for a User
```bash
curl "http://localhost:8000/api/v1/matching/user-123/matches?top_k=10&similarity_threshold=0.3"
```

#### Response Format
```json
{
  "success": true,
  "user_id": "user-123",
  "total_matches": 5,
  "requirements_matches": [
    {
      "user_id": "user-456", 
      "similarity_score": 0.85,
      "match_type": "requirements_to_offerings",
      "explanation": "Excellent match (0.85). Your requirements align well with their offerings."
    }
  ],
  "offerings_matches": [
    {
      "user_id": "user-789",
      "similarity_score": 0.72, 
      "match_type": "offerings_to_requirements",
      "explanation": "Strong match (0.72). Your offerings match what they're looking for."
    }
  ],
  "message": "Found 5 total matches"
}
```

### Performance Features

- **Vector Indexing**: PostgreSQL pgvector indexes for fast similarity search
- **Caching**: In-memory embedding cache for frequently accessed data
- **Async Processing**: Background embedding generation doesn't block API
- **Configurable Limits**: Adjustable result limits and similarity thresholds
- **CPU-Optimized**: Uses efficient CPU-only models for broad compatibility

## Data Persistence

**Data persistence behavior:**

- **Redis data** persists across restarts using Docker named volume `redis_data`
- **PostgreSQL data** persists across restarts using Docker named volume `pgdata`
- **DynamoDB tables and data** persist across restarts using Docker named volume `localstack_data`
- **S3 buckets and files** do NOT persist (limitation of free LocalStack version)
- All data survives `docker compose down` and `docker compose up -d`
- Configured with named volumes in `docker-compose.yml`

**Important:** Free LocalStack does not support S3 persistence. S3 buckets and files must be recreated after container restarts.

**To reset all data:**
```bash
docker compose down
docker volume rm reciprocity-ai_localstack_data reciprocity-ai_redis_data reciprocity-ai_pgdata
docker compose up -d
```

**To view volume data:**
```bash
# View all volumes
docker volume ls | grep reciprocity-ai

# Inspect specific volumes
docker volume inspect reciprocity-ai_localstack_data
docker volume inspect reciprocity-ai_redis_data
docker volume inspect reciprocity-ai_pgdata
```

**To test persistence:**
Check that data survives container restarts by stopping and starting services.

## Development Workflow

1. **Create DynamoDB table**: Use the AWS CLI command from [Table Creation](#table-creation) section
2. **Create S3 bucket** (optional): Use the S3 commands above to create `reciprocity-resumes` bucket and upload test resumes
3. **Copy environment variables**: `cp env.example .env` and adjust as needed
4. **Start development**: `./dev.sh` (starts Redis, PostgreSQL, and LocalStack with persistence, FastAPI server, and Celery worker)
5. **Test API**: Use curl commands above (basic test or S3 URL test)
6. **Check logs**: 
   - FastAPI server logs: Check the terminal output
   - Celery worker logs: Check the terminal output for task processing
7. **Stop**: Press Ctrl+C to stop all services gracefully (stops FastAPI, Celery, Redis, PostgreSQL, and LocalStack)

## Testing Resume Processing

For testing resume processing with actual files:

1. **Create S3 bucket**: `aws --endpoint-url=http://localhost:4566 s3 mb s3://reciprocity-resumes`
2. **Upload resume**: `aws --endpoint-url=http://localhost:4566 s3 cp "/path/to/resume.pdf" s3://reciprocity-resumes/resumes/`
3. **Test with S3 URL**: Use the LocalStack S3 URL in your API test

The system will download the resume from the S3 URL, parse it using LangChain, and store the extracted text in DynamoDB.

## Testing Matchmaking System

For testing the complete matchmaking pipeline:

### 1. Register Multiple Users
```bash
# Register first user (developer without resume)
curl -X POST "http://localhost:8000/api/v1/user/register" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "developer-001",
    "questions": [
      {
        "prompt": "What are you looking for?",
        "answer": "I need funding and mentorship for my AI startup"
      },
      {
        "prompt": "What can you offer?",
        "answer": "Full-stack development expertise and AI/ML implementation"
      },
      {
        "prompt": "What is your experience level?",
        "answer": "5+ years in software development, 2 years in AI/ML"
      }
    ]
  }'

# Register second user (investor with resume)
curl -X POST "http://localhost:8000/api/v1/user/register" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "investor-001", 
    "resume_link": "https://example.com/investor-resume.pdf",
    "questions": [
      {
        "prompt": "What are you looking for?",
        "answer": "Technical co-founders with AI expertise for portfolio companies"
      },
      {
        "prompt": "What can you offer?",
        "answer": "Seed funding, business mentorship, and network connections"
      }
    ]
  }'
```

### 2. Wait for Processing
The system will automatically execute a Celery task chain:
1. **Resume Processing**: Parse resume and extract text, or skip if no resume (5-30 seconds)
2. **Persona Generation**: Generate AI persona with requirements/offerings (30-60 seconds)  
3. **Embedding Generation**: Create semantic embeddings, store in PostgreSQL, and send matches notification (10-20 seconds)

Each step runs automatically after the previous completes. Backend notifications are sent automatically.

### 3. Test Matching
```bash
# Get all matches for the developer
curl "http://localhost:8000/api/v1/matching/developer-001/matches?top_k=10&similarity_threshold=0.2"

# Get specific match types
curl "http://localhost:8000/api/v1/matching/developer-001/requirements-matches?top_k=5"
curl "http://localhost:8000/api/v1/matching/developer-001/offerings-matches?top_k=5"

# Check matching statistics
curl "http://localhost:8000/api/v1/matching/stats"
```

### 4. Run Tests

#### Complete End-to-End Test
```bash
# Execute complete end-to-end test
uv run python tests/test_end_to_end.py
```

The test validates:
- API health and connectivity
- User registration without resume (optional resume feature)
- Celery task chain execution and completion
- Persona generation from questions only
- Embedding generation and storage in PostgreSQL
- Matchmaking functionality with similarity scoring
- All API endpoints and error handling

#### Notification System Test
```bash
# Set your backend URL first
export RECIPROCITY_BACKEND_URL=http://localhost:3000/api

# Test backend notification system
uv run python tests/test_matches_notification.py
```

This test validates:
- User registration and processing pipeline completion
- Persona generation and backend notification calls
- Embedding generation and matches notification calls
- Actual backend endpoints: `/user/summary-ready` and `/user-matches-ready`
- Complete workflow from registration to matches notification

**Note**: This test calls your actual backend. Make sure your backend is running and can receive the notifications.

## User Feedback System

The platform includes a feedback system that allows users to provide input on matches and chats, which is used to continuously improve the matching algorithm.

### How Feedback Works

1. **Submit Feedback**: Users provide feedback via the `/api/v1/user/feedback` endpoint
2. **Store Feedback**: Feedback is stored in DynamoDB (`user_feedback` table)
3. **Update Embeddings**: The user's persona embeddings are automatically adjusted based on sentiment:
   - **Positive feedback** (contains "good", "great", "excellent", etc.): Embeddings boosted by 5%
   - **Negative feedback**: Embeddings reduced by 5%
4. **Improved Matching**: Future matches reflect the user's preferences based on historical feedback

### Feedback API Example

```bash
curl -X POST "http://localhost:8000/api/v1/user/feedback" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key" \
  -d '{
    "user_id": "developer-001",
    "type": "match",
    "feedback_id": "550e8400-e29b-41d4-a716-446655440000",
    "feedback": "Great match! Very relevant to my interests."
  }'
```

**Request Fields:**
- `user_id` (string): The user providing feedback
- `type` (string): Either "match" or "chat"
- `feedback_id` (string): The match or chat ID being reviewed
- `feedback` (string): The user's feedback text

**Response:**
```json
{
  "Code": 200,
  "Message": "success",
  "Result": {
    "success": true,
    "message": "Feedback saved and persona updated",
    "data": {
      "user_id": "developer-001",
      "type": "match",
      "feedback_id": "550e8400-e29b-41d4-a716-446655440000",
      "feedback": "Great match! Very relevant to my interests."
    }
  }
}
```

### Future Improvements

The current implementation uses simple sentiment analysis and vector scaling. Future versions can incorporate:
- Advanced NLP for better sentiment understanding
- Specific feature extraction from feedback
- Personalized weighting based on feedback history
- Integration with chat data for context-aware adjustments

## Troubleshooting

### Celery on Windows (--pool=solo Required)

**Problem:** Celery fails with `fork()` errors or workers hang on Windows.

**Root Cause:** Celery's default `prefork` pool uses `fork()` which is not supported on Windows.

**Solution:** Always use `--pool=solo` when running Celery on Windows:

```bash
# Windows: Start Celery worker
.venv\Scripts\celery -A app.core.celery worker --pool=solo -l info

# Or use the provided startup script:
scripts\start_celery_worker.bat
```

**Why `--pool=solo`:**
- Windows doesn't support POSIX `fork()` system call
- `--pool=solo` runs tasks sequentially in the main process
- Production deployments on Linux/Mac can use `--pool=prefork` for concurrency

**Batch Script:** A convenience script is provided at `scripts/start_celery_worker.bat`:
```batch
@echo off
cd /d %~dp0..
call .venv\Scripts\activate.bat
celery -A app.core.celery worker --pool=solo -l info
pause
```

### LocalStack Data Not Persisting

**Problem:** Tables and S3 buckets disappear after restarting LocalStack.

**Solution:**
```bash
# Check if persistence is enabled
docker compose logs localstack | grep -i persistence

# If not working, reset and restart with persistence
docker compose down
rm -rf ./localstack-data/
docker compose up -d

# Test persistence
uv run python tests/test_localstack_persistence.py
```

**Why this happens:**
- LocalStack persistence requires `PERSISTENCE=1` environment variable
- Volume mount `./localstack-data:/var/lib/localstack` must be configured
- Data is stored in `./localstack-data/state/` directory

### Services Not Starting

If you see connection errors for Redis, PostgreSQL, or LocalStack:

**Solution:**
```bash
# Stop all services and restart
docker compose down
docker compose up -d

# Wait for services to be ready
sleep 10

# Then start development environment
./dev.sh
```

**Why this happens:**
- Services may take time to initialize
- Docker containers need to be fully ready before connections work
- The dev.sh script includes health checks for all services

## Production Deployment

### Environment Configuration

**Production environment variables (create `.env.production`):**

```bash
# Application Configuration
APP_NAME=reciprocity-platform
APP_VERSION=1.0.0
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8000
HOST=0.0.0.0

# CORS Configuration (update with your frontend domain)
CORS_ORIGINS=["https://your-frontend-domain.com"]
ALLOWED_HOSTS=["your-api-domain.com"]

# Database Configuration
REDIS_URL=redis://your-redis-host:6379/0

# AWS Configuration (Production - MINIMAL!)
DYNAMO_PROFILE_TABLE_NAME=user_profiles_prod
# That's it! AWS auto-detects region, credentials, and endpoints

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-your-production-openai-key
OPENAI_MODEL=gpt-4o-mini

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.3

# PostgreSQL Configuration (Production)
DATABASE_URL=postgresql://user:password@your-postgres-host:5432/reciprocity_ai

# Backend Integration
RECIPROCITY_BACKEND_URL=https://your-backend-domain.com/api
```

### Prerequisites

- EC2 instance with IAM role that has DynamoDB permissions
- DynamoDB table created (e.g., `user_profiles_prod`)
- PostgreSQL and Redis databases accessible
- Production environment file configured

### Production Deployment Steps

**1. Server Setup:**
```bash
# Clone repository
git clone <your-repo-url>
cd reciprocity-ai

# Install uv (fast Python package manager)
pip install uv

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Create production environment file
cp env.production.example .env
# Edit .env with your actual production values
```

**2. Database Setup:**
```bash
# Create DynamoDB tables (production)
aws dynamodb create-table \
  --table-name user_profiles_prod \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

aws dynamodb create-table \
  --table-name user_matches_prod \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

aws dynamodb create-table \
  --table-name user_feedback_prod \
  --attribute-definitions AttributeName=feedback_id,AttributeType=S \
  --key-schema AttributeName=feedback_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

aws dynamodb create-table \
  --table-name ai_chat_records_prod \
  --attribute-definitions AttributeName=chat_id,AttributeType=S \
  --key-schema AttributeName=chat_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# Run PostgreSQL migrations
uv run alembic upgrade head
```

**3. Start Production Services:**

**Option A: Manual Start (for testing)**
```bash
# Terminal 1: Start API server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Celery worker + beat scheduler
uv run celery -A app.core.celery worker --beat --loglevel=info

# OR run worker and beat separately:
# Terminal 2: Start Celery worker
# uv run celery -A app.core.celery worker --loglevel=info

# Terminal 3: Start Celery beat scheduler
# uv run celery -A app.core.celery beat --loglevel=info
```

**Option B: Production Start (recommended)**
```bash
# Start API server in background
nohup uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

# Start Celery worker + beat scheduler in background  
nohup uv run celery -A app.core.celery worker --beat --loglevel=info > celery.log 2>&1 &

# OR run them separately:
# nohup uv run celery -A app.core.celery worker --loglevel=info > celery-worker.log 2>&1 &
# nohup uv run celery -A app.core.celery beat --loglevel=info > celery-beat.log 2>&1 &

# Check processes are running
ps aux | grep uvicorn
ps aux | grep celery
```

**Option C: Using Process Manager (systemd)**

Create `/etc/systemd/system/reciprocity-api.service`:
```ini
[Unit]
Description=Reciprocity AI API Server
After=network.target

[Service]
Type=exec
User=ubuntu
WorkingDirectory=/home/ubuntu/reciprocity-ai
ExecStart=/usr/local/bin/uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/reciprocity-worker.service`:
```ini
[Unit]
Description=Reciprocity AI Celery Worker and Beat Scheduler
After=network.target

[Service]
Type=exec
User=ubuntu
WorkingDirectory=/home/ubuntu/reciprocity-ai
ExecStart=/usr/local/bin/uv run celery -A app.core.celery worker --beat --loglevel=info
EnvironmentFile=/home/ubuntu/reciprocity-ai/.env.production
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Note:** The `--beat` flag runs the beat scheduler alongside the worker. For production, you can also run them separately by creating two service files.

```bash
# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable reciprocity-api reciprocity-worker
sudo systemctl start reciprocity-api reciprocity-worker

# Check status
sudo systemctl status reciprocity-api
sudo systemctl status reciprocity-worker
```

### Production Checklist

**Services Running:**
- [ ] API server running (uvicorn on port 8000)
- [ ] Celery worker running
- [ ] Both services auto-restart on failure
- [ ] Health check endpoint responding

**Configuration:**
- [ ] Production environment file configured
- [ ] Database connections working
- [ ] DynamoDB table accessible
- [ ] OpenAI API key working

### Testing Production Setup

```bash
# Check API health
curl https://your-domain.com/health

# Test user registration
curl -X POST https://your-domain.com/api/v1/user/register \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_prod_001",
    "questions": [
      {
        "prompt": "What are you looking for?",
        "answer": "Testing production deployment"
      }
    ]
  }'
```

### Production Monitoring

**Service Health Checks:**
```bash
# Check API server
curl https://your-domain.com/health

# Check service status
sudo systemctl status reciprocity-api reciprocity-worker

# Check logs
sudo journalctl -u reciprocity-api -f
sudo journalctl -u reciprocity-worker -f

# Check processes
ps aux | grep -E "(uvicorn|celery)"
```

**Key Metrics to Monitor:**
- **API**: Response times, error rates, uptime
- **DynamoDB**: Read/write capacity, throttling, item counts
- **PostgreSQL**: Connection pool usage, query performance
- **Celery**: Task queue length, processing times, failed tasks
- **OpenAI**: API usage, rate limits, costs
- **System**: CPU, memory, disk usage

**Log Locations:**
- **API logs**: `sudo journalctl -u reciprocity-api`
- **Worker logs**: `sudo journalctl -u reciprocity-worker`
- **Application logs**: Check LOG_LEVEL in environment

## Project Structure

```
reciprocity-ai/
├── app/                    # FastAPI application
│   ├── routers/           # API route handlers
│   │   ├── user.py       # User registration and profiles
│   │   ├── matching.py   # Matchmaking endpoints
│   │   └── health.py     # Health check endpoints
│   ├── services/          # Business logic services
│   │   ├── embedding_service.py      # SentenceTransformers + pgvector
│   │   ├── match_explanation_service.py  # Match explanations
│   │   ├── filtering_service.py      # Advanced filtering
│   │   └── persona_service.py        # AI persona generation
│   ├── workers/           # Background Celery tasks
│   │   ├── embedding_processing.py   # Embedding generation
│   │   ├── persona_processing.py     # Persona generation
│   │   └── resume_processing.py      # Resume parsing
│   ├── adapters/          # Data access layer
│   │   ├── dynamodb.py   # DynamoDB operations
│   │   └── postgresql.py # PostgreSQL + pgvector operations
│   ├── schemas/           # Pydantic models
│   ├── core/              # Core configuration
│   │   └── celery.py      # Celery configuration
│   └── main.py           # FastAPI app
├── alembic/               # Database migrations
├── scripts/               # Development scripts
│   ├── start_server.py   # Start FastAPI server
│   └── start_celery.py   # Start Celery worker
├── tests/                 # Test files
├── docker-compose.yml     # PostgreSQL, Redis, and LocalStack configuration
├── dev.sh                # Quick development setup
└── README.md             # This file

## Backend Integration

The service automatically notifies your backend when persona generation completes.

### Configuration
Set `RECIPROCITY_BACKEND_URL` in your `.env` file:
```bash
RECIPROCITY_BACKEND_URL=http://localhost:3000/api
```

### Notification Flow
1. User completes registration (resume is optional)
2. Celery task chain executes: Resume → Persona → Embeddings
3. Resume processing completes (or skips if no resume provided)
4. AI persona generation completes
5. **Persona ready notification** sent to `{RECIPROCITY_BACKEND_URL}/user/summary-ready`
6. Embedding generation completes
7. **Matches ready notification** sent to `{RECIPROCITY_BACKEND_URL}/user/matches-ready` (always sent, even with empty matches)

### Payload Formats

#### Persona Ready Notification
```json
{
  "user_id": "user-123",
  "persona": {
    "name": "The AI-Driven Integrator",
    "archetype": "Tech-Savvy Full Stack Innovator",
    "experience": "4+ Years in Full Stack Development with AI and Crypto Integration",
    "focus": "AI-Powered Products | Crypto & Blockchain Platforms | Scalable Marketplaces | Real-Time Systems",
    "profile_essence": "The AI-Driven Integrator combines deep technical expertise...",
    "investment_philosophy": "- Invests in startups leveraging AI and machine learning...",
    "what_theyre_looking_for": "Early to growth-stage startups focused on AI-driven SaaS...",
    "engagement_style": "Engages actively through collaborative, cross-functional partnerships...",
    "generated_at": "2025-09-27T15:13:02.420472+00:00"
  }
}
```

#### Matches Ready Notification
```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user-123",
  "matches": [
    {
      "target_user_id": "user-456"
    },
    {
      "target_user_id": "user-789"
    }
  ]
}
```

### Backend Endpoints
Your backend should implement:

#### Persona Ready Endpoint
- **Endpoint**: `POST /user/summary-ready`
- **Content-Type**: `application/json`
- **Response**: Any valid JSON (logged by the service)

#### Matches Ready Endpoint
- **Endpoint**: `POST /user/matches-ready`
- **Content-Type**: `application/json`
- **Response**: `{"success": true, "message": "matches received!"}`
- **Description**: Receives user's requirements matches (what user needs vs others' offerings)
- **Note**: Always called after embedding generation, even with empty matches array
```
# Trigger redeploy Thu, Feb 26, 2026  2:17:01 AM
