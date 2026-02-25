"""
Initialize DynamoDB tables and process all existing users.
This script:
1. Creates the DynamoDB tables if they don't exist
2. Fetches all users from PostgreSQL backend
3. Sends webhook notification for each user

Usage:
    cd reciprocity-ai
    .venv/Scripts/python.exe init_and_process.py
"""
import os
import sys
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv(override=True)

import boto3
from botocore.exceptions import ClientError

# DynamoDB setup
DYNAMODB_ENDPOINT = os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566')
REGION = 'us-east-1'

print("=" * 60)
print("RECIPROCITY AI - INITIALIZE AND PROCESS")
print("=" * 60)

# Create DynamoDB resource
ddb = boto3.resource(
    'dynamodb',
    endpoint_url=DYNAMODB_ENDPOINT,
    region_name=REGION,
    aws_access_key_id='test',
    aws_secret_access_key='test'
)

client = boto3.client(
    'dynamodb',
    endpoint_url=DYNAMODB_ENDPOINT,
    region_name=REGION,
    aws_access_key_id='test',
    aws_secret_access_key='test'
)

def create_table_if_not_exists(table_name: str, key_schema: list, attr_defs: list):
    """Create a DynamoDB table if it doesn't exist."""
    try:
        existing = client.list_tables()['TableNames']
        if table_name in existing:
            print(f"  [OK] Table '{table_name}' already exists")
            return True
    except Exception as e:
        print(f"  [!] Error checking tables: {e}")

    try:
        table = ddb.create_table(
            TableName=table_name,
            KeySchema=key_schema,
            AttributeDefinitions=attr_defs,
            BillingMode='PAY_PER_REQUEST'
        )
        table.wait_until_exists()
        print(f"  [OK] Created table '{table_name}'")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            print(f"  [OK] Table '{table_name}' already exists")
            return True
        print(f"  [FAIL] Error creating table '{table_name}': {e}")
        return False

print("\n[1] Creating DynamoDB tables...")

# Create main tables
tables = [
    {
        'name': os.getenv('DYNAMO_PROFILE_TABLE_NAME', 'reciprocity-profiles'),
        'key': [{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
        'attrs': [{'AttributeName': 'user_id', 'AttributeType': 'S'}]
    },
    {
        'name': 'user_matches',
        'key': [{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
        'attrs': [{'AttributeName': 'user_id', 'AttributeType': 'S'}]
    },
    {
        'name': 'user_feedback',
        'key': [{'AttributeName': 'feedback_id', 'KeyType': 'HASH'}],
        'attrs': [{'AttributeName': 'feedback_id', 'AttributeType': 'S'}]
    },
    {
        'name': 'notified_match_pairs',
        'key': [{'AttributeName': 'pair_key', 'KeyType': 'HASH'}],
        'attrs': [{'AttributeName': 'pair_key', 'AttributeType': 'S'}]
    },
    {
        'name': 'ai_chat_records',
        'key': [{'AttributeName': 'chat_id', 'KeyType': 'HASH'}],
        'attrs': [{'AttributeName': 'chat_id', 'AttributeType': 'S'}]
    },
]

for t in tables:
    create_table_if_not_exists(t['name'], t['key'], t['attrs'])

# Verify tables
print("\n[2] Verifying tables...")
existing_tables = client.list_tables()['TableNames']
print(f"  Tables in DynamoDB: {existing_tables}")

# Fetch users from backend webhook
print("\n[3] Fetching users from backend...")
BACKEND_URL = os.getenv('RECIPROCITY_BACKEND_URL', 'http://localhost:3000')
WEBHOOK_API_KEY = os.getenv('WEBHOOK_API_KEY', 'dev-webhook-key')

headers = {
    "Content-Type": "application/json",
    "x-api-key": WEBHOOK_API_KEY
}

try:
    resp = requests.get(
        f"{BACKEND_URL}/api/v1/webhooks/list-users",
        headers=headers,
        timeout=30
    )
    if resp.status_code == 200:
        data = resp.json()
        result = data.get('result', {})
        users = result.get('rows', [])
        print(f"  [OK] Found {len(users)} users in backend")
    else:
        print(f"  [FAIL] Could not fetch users: {resp.status_code} - {resp.text[:100]}")
        users = []
except Exception as e:
    print(f"  [ERROR] {e}")
    users = []

if not users:
    print("\nNo users found. Exiting.")
    sys.exit(0)

# For each user, fetch their data and trigger processing
print("\n[4] Processing users...")

# Import pynamodb models
from app.adapters.dynamodb import UserProfile
from app.services.notification_service import NotificationService

notification_service = NotificationService()
processed = 0
failed = 0

for i, user in enumerate(users, 1):
    user_id = str(user.get('id'))
    email = user.get('email', 'N/A')
    name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or 'N/A'

    print(f"\n  [{i}/{len(users)}] {name} ({email})")

    try:
        # Get user data from backend
        user_resp = requests.get(
            f"{BACKEND_URL}/api/v1/webhooks/get-user-data?user_id={user_id}",
            headers=headers,
            timeout=30
        )

        if user_resp.status_code != 200:
            print(f"    [!] Failed to get user data: {user_resp.status_code}")
            failed += 1
            continue

        user_data = user_resp.json()
        questions = user_data.get('questions', [])
        resume_link = user_data.get('resume_link')

        if not questions:
            print(f"    [!] No questions found, skipping")
            failed += 1
            continue

        # Check if user already exists in DynamoDB
        try:
            existing = UserProfile.get(user_id)
            if existing.persona_status == 'completed':
                print(f"    [OK] Already has persona: {existing.persona.name}")
                # Send webhook notification to sync with backend
                if notification_service.is_configured():
                    notification_service.send_persona_ready_notification(user_id)
                processed += 1
                continue
        except UserProfile.DoesNotExist:
            pass

        # Create user profile in DynamoDB
        profile = UserProfile.create_user(user_id, resume_link, questions)
        profile.save()
        print(f"    [OK] Created profile in DynamoDB ({len(questions)} questions)")

        processed += 1

    except Exception as e:
        print(f"    [ERROR] {e}")
        failed += 1

print("\n" + "=" * 60)
print(f"SUMMARY: {processed} processed, {failed} failed")
print("=" * 60)

print("\n[5] Next steps:")
print("  1. Start Celery worker: .venv\\Scripts\\celery -A app.core.celery worker --pool=solo -l info")
print("  2. Trigger persona generation for all users via API")
print("  3. Run complete_existing_users.py to finish the flow")
