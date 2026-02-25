#!/usr/bin/env python
"""
Initialize DynamoDB tables for Reciprocity AI.
Run this after LocalStack container starts to ensure tables exist.

Usage:
    python scripts/init_dynamodb.py
"""
import os
import sys
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv(override=True)

DYNAMODB_ENDPOINT = os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566')
REGION = os.getenv('AWS_REGION', 'us-east-1')

print("=" * 60)
print("DYNAMODB TABLE INITIALIZATION")
print("=" * 60)
print(f"Endpoint: {DYNAMODB_ENDPOINT}")

# Create DynamoDB client
client = boto3.client(
    'dynamodb',
    endpoint_url=DYNAMODB_ENDPOINT,
    region_name=REGION,
    aws_access_key_id='test',
    aws_secret_access_key='test'
)

# Table definitions
TABLES = [
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


def create_table_if_not_exists(table_config: dict) -> bool:
    """Create a DynamoDB table if it doesn't exist."""
    table_name = table_config['name']

    try:
        # Check if table exists
        existing = client.list_tables()['TableNames']
        if table_name in existing:
            print(f"  [OK] {table_name} already exists")
            return True
    except Exception as e:
        print(f"  [!] Error checking tables: {e}")

    try:
        # Create table
        resource = boto3.resource(
            'dynamodb',
            endpoint_url=DYNAMODB_ENDPOINT,
            region_name=REGION,
            aws_access_key_id='test',
            aws_secret_access_key='test'
        )

        table = resource.create_table(
            TableName=table_name,
            KeySchema=table_config['key'],
            AttributeDefinitions=table_config['attrs'],
            BillingMode='PAY_PER_REQUEST'
        )
        table.wait_until_exists()
        print(f"  [OK] Created {table_name}")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            print(f"  [OK] {table_name} already exists")
            return True
        print(f"  [FAIL] Error creating {table_name}: {e}")
        return False


def main():
    print("\nCreating tables...")

    success = 0
    for table in TABLES:
        if create_table_if_not_exists(table):
            success += 1

    print(f"\nResult: {success}/{len(TABLES)} tables ready")

    # List all tables
    existing = client.list_tables()['TableNames']
    print(f"Tables in DynamoDB: {existing}")

    return success == len(TABLES)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
