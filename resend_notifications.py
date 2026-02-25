"""
Resend webhook notifications for all completed personas.
Uses the corrected WEBHOOK_API_KEY.
"""
import os
import sys
import requests
from dotenv import load_dotenv

# Force reload .env to pick up updated WEBHOOK_API_KEY
load_dotenv(override=True)

# Configuration
BACKEND_URL = os.getenv('RECIPROCITY_BACKEND_URL', 'http://localhost:3000')
WEBHOOK_API_KEY = os.getenv('WEBHOOK_API_KEY', 'dev-webhook-key')
WEBHOOK_ENDPOINT = f"{BACKEND_URL}/api/v1/webhooks/summary-ready"

print(f"Backend URL: {BACKEND_URL}")
print(f"Webhook Key: {WEBHOOK_API_KEY}")
print(f"Webhook Endpoint: {WEBHOOK_ENDPOINT}")

# Get all user IDs from DynamoDB
import boto3
from boto3.dynamodb.conditions import Attr

dynamodb = boto3.resource(
    'dynamodb',
    endpoint_url=os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566'),
    region_name='us-east-1',
    aws_access_key_id='test',
    aws_secret_access_key='test'
)

table = dynamodb.Table('reciprocity-profiles')

# Scan for all users with completed persona status
response = table.scan(
    FilterExpression=Attr('persona_status').eq('completed')
)

users = response.get('Items', [])
print(f"\nFound {len(users)} users with completed personas")

# Send webhook for each user
success_count = 0
fail_count = 0

for user in users:
    user_id = user.get('user_id')
    persona = user.get('persona', {})

    if not user_id:
        continue

    payload = {
        "userId": user_id,
        "status": "completed",
        "personaName": persona.get('name', 'Unknown'),
        "message": "AI persona generation completed successfully"
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": WEBHOOK_API_KEY
    }

    try:
        resp = requests.post(WEBHOOK_ENDPOINT, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200 or resp.status_code == 201:
            print(f"  [OK] {user_id}: {persona.get('name', 'N/A')}")
            success_count += 1
        else:
            print(f"  [FAIL] {user_id}: {resp.status_code} - {resp.text[:100]}")
            fail_count += 1
    except Exception as e:
        print(f"  [ERROR] {user_id}: {str(e)}")
        fail_count += 1

print(f"\nResults: {success_count} succeeded, {fail_count} failed")
