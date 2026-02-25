#!/usr/bin/env python3
"""
End-to-End Verification Script for Reciprocity AI Platform
Tests the complete user journey from onboarding to matching.
"""
import os
import sys
import json
import uuid
import requests
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv()

BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "dev-api-key-change-in-production")
HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY": API_KEY
}

def print_result(test_name: str, passed: bool, details: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} | {test_name}")
    if details:
        print(f"         {details}")

def test_health():
    """Test 1: Health Check"""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        data = resp.json()
        passed = resp.status_code == 200 and data.get("success") == True
        print_result("Health Check", passed, f"Status: {data.get('data', {}).get('status')}")
        return passed
    except Exception as e:
        print_result("Health Check", False, str(e))
        return False

def test_prediction():
    """Test 2: Answer Prediction"""
    try:
        # Test exact match prediction
        payload = {
            "options": [
                {"label": "Networking", "value": "networking"},
                {"label": "Job Search", "value": "job_search"},
                {"label": "Mentorship", "value": "mentorship"},
                {"label": "Collaboration", "value": "collaboration"}
            ],
            "user_response": "mentorship"
        }
        resp = requests.post(f"{BASE_URL}/api/v1/predict-answer", json=payload, headers=HEADERS, timeout=10)
        data = resp.json()
        passed = resp.status_code == 200 and data.get("predicted_answer") == "Mentorship"
        print_result("Answer Prediction", passed, f"Predicted: {data.get('predicted_answer')}, Valid: {data.get('valid_answer')}")
        return passed
    except Exception as e:
        print_result("Answer Prediction", False, str(e))
        return False

def test_question_modification():
    """Test 3: Question Tone Modification (requires OpenAI)"""
    try:
        payload = {
            "question_id": "q_goals",
            "code": "GOALS_001",
            "prompt": "What are your professional goals?",
            "suggestion_chips": "Career growth,Learning,Leadership",
            "previous_user_response": []
        }
        resp = requests.post(f"{BASE_URL}/api/v1/modify-question", json=payload, headers=HEADERS, timeout=45)

        if resp.status_code == 200:
            data = resp.json()
            passed = "ai_text" in data and len(data.get("ai_text", "")) > 20
            ai_preview = data.get("ai_text", "")[:60] + "..." if len(data.get("ai_text", "")) > 60 else data.get("ai_text", "")
            print_result("Question Modification (OpenAI)", passed, f"AI: {ai_preview}")
            return passed
        elif resp.status_code == 500:
            print_result("Question Modification (OpenAI)", False, "OpenAI API error")
            return False
        else:
            print_result("Question Modification (OpenAI)", False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        print_result("Question Modification (OpenAI)", False, str(e))
        return False

def test_validation():
    """Test 4: Input Validation"""
    try:
        # Test with invalid data
        payload = {
            "options": [],  # Empty - should fail validation
            "user_response": "test"
        }
        resp = requests.post(f"{BASE_URL}/api/v1/predict-answer", json=payload, headers=HEADERS, timeout=5)
        passed = resp.status_code == 422  # Validation error
        print_result("Input Validation", passed, f"Empty options correctly rejected with {resp.status_code}")
        return passed
    except Exception as e:
        print_result("Input Validation", False, str(e))
        return False

def test_auth():
    """Test 5: API Key Authentication"""
    try:
        bad_headers = {"Content-Type": "application/json", "X-API-KEY": "wrong-key"}
        payload = {"options": [{"label": "A", "value": "a"}], "user_response": "A"}
        resp = requests.post(f"{BASE_URL}/api/v1/predict-answer", json=payload, headers=bad_headers, timeout=5)
        passed = resp.status_code == 403  # Forbidden
        print_result("API Key Auth", passed, f"Invalid key rejected with {resp.status_code}")
        return passed
    except Exception as e:
        print_result("API Key Auth", False, str(e))
        return False

def test_redis_cache():
    """Test 6: Redis Cache Integration"""
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6380/0"))
        r.ping()
        keys = r.dbsize()
        print_result("Redis Cache", True, f"Connected, {keys} keys cached")
        return True
    except Exception as e:
        print_result("Redis Cache", False, str(e))
        return False

def test_postgres():
    """Test 7: PostgreSQL + pgvector"""
    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        version = cur.fetchone()
        conn.close()
        passed = version is not None
        print_result("PostgreSQL + pgvector", passed, f"pgvector version: {version[0] if version else 'not installed'}")
        return passed
    except Exception as e:
        print_result("PostgreSQL + pgvector", False, str(e))
        return False

def test_dynamodb():
    """Test 8: DynamoDB (LocalStack)"""
    try:
        import boto3
        dynamodb = boto3.client(
            'dynamodb',
            endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test")
        )
        tables = dynamodb.list_tables()['TableNames']
        passed = len(tables) >= 2
        print_result("DynamoDB (LocalStack)", passed, f"Tables: {', '.join(tables)}")
        return passed
    except Exception as e:
        print_result("DynamoDB (LocalStack)", False, str(e))
        return False

def main():
    print("\n" + "="*60)
    print("  RECIPROCITY AI - End-to-End Verification")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    results = []

    print("--- API Endpoints ---")
    results.append(test_health())
    results.append(test_prediction())
    results.append(test_question_modification())
    results.append(test_validation())
    results.append(test_auth())

    print("\n--- Infrastructure ---")
    results.append(test_redis_cache())
    results.append(test_postgres())
    results.append(test_dynamodb())

    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"  RESULTS: {passed}/{total} tests passed ({100*passed//total}%)")

    if passed == total:
        print("  STATUS: ALL SYSTEMS OPERATIONAL")
    else:
        print("  STATUS: SOME ISSUES DETECTED")
    print("="*60 + "\n")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
