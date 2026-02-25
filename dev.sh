#!/bin/bash

echo "Starting Reciprocity AI Development Environment"
echo "WARNING: This is for LOCAL DEVELOPMENT ONLY"
echo "Production uses real AWS DynamoDB (tables created via infrastructure)"
echo ""

if ! curl -s http://localhost:4566/_localstack/health > /dev/null 2>&1; then
    echo "Starting services (PostgreSQL + Redis + LocalStack)..."
    docker compose up -d
    echo "Waiting for services to be ready..."
    sleep 10
else
    echo "Services are already running"
fi

# Load environment variables first
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "No .env file found, loading from env.example"
    export $(cat env.example | grep -v '^#' | xargs)
fi

echo ""
echo "Checking Redis connection..."
if docker exec reciprocity-redis redis-cli ping > /dev/null 2>&1; then
    echo "Redis is ready"
else
    echo "Redis is not ready yet, waiting..."
    sleep 3
fi

echo ""
echo "Checking PostgreSQL connection..."
if docker exec reciprocity-postgres pg_isready -U reciprocity_user -d reciprocity_ai > /dev/null 2>&1; then
    echo "PostgreSQL is ready"
    
    echo "Running database migrations..."
    if uv run alembic upgrade head > /dev/null 2>&1; then
        echo "Database migrations completed"
    else
        echo "Database migrations failed or already up to date"
    fi
else
    echo "PostgreSQL is not ready yet, waiting..."
    sleep 3
fi

echo ""
echo "Checking if DynamoDB table exists..."
if aws dynamodb describe-table --table-name user_profiles --endpoint-url http://localhost:4566 --region us-east-1 > /dev/null 2>&1; then
    echo "Table 'user_profiles' exists"
else
    echo "Table 'user_profiles' does not exist"
fi

# Check for required environment variables
if [ -z "$PORT" ] || [ -z "$HOST" ]; then
    echo "ERROR: Missing required environment variables: PORT, HOST"
    echo "Please set these in your .env file or environment"
    exit 1
fi

echo ""
echo "Starting FastAPI server and Celery worker..."
echo "Server will be available at: http://localhost:$PORT"
echo "API docs will be available at: http://localhost:$PORT/docs"
echo ""
echo "Starting services in background..."
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Stopping services..."
    
    # Kill background processes if they exist
    if [ ! -z "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "Stopping FastAPI server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    
    if [ ! -z "$CELERY_PID" ] && kill -0 $CELERY_PID 2>/dev/null; then
        echo "Stopping Celery worker (PID: $CELERY_PID)..."
        kill $CELERY_PID 2>/dev/null
        wait $CELERY_PID 2>/dev/null
    fi
    
    # Stop Docker Compose services (PostgreSQL, Redis, LocalStack)
    echo "Stopping Docker services..."
    docker compose down
    
    echo "All services stopped."
    exit 0
}

# Set up signal handlers
trap cleanup INT TERM

# Start FastAPI server in background
echo "Starting FastAPI server..."
uv run python scripts/start_server.py &
SERVER_PID=$!

# Start Celery worker in background
echo "Starting Celery worker..."
uv run python scripts/start_celery.py &
CELERY_PID=$!

echo ""
echo "Services started:"
echo "- FastAPI server (PID: $SERVER_PID)"
echo "- Celery worker (PID: $CELERY_PID)"
echo ""
echo "To check logs:"
echo "- Server logs: Check terminal output above"
echo "- Celery logs: Check terminal output above"
echo ""
echo "To stop all services:"
echo "- Press Ctrl+C to stop this script"
echo "- Or kill processes: kill $SERVER_PID $CELERY_PID"
echo ""
echo "Press Ctrl+C to stop all services..."

# Keep script running and wait for background processes
wait
