"""
Direct User Sync Script
=======================
Bypasses the broken backend webhook endpoint by directly querying PostgreSQL
for user data, creating DynamoDB profiles, and triggering persona generation.

Usage:
    cd reciprocity-ai
    .venv/Scripts/python.exe sync_users_direct.py
"""
import os
import sys
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(override=True)

print("=" * 60)
print("DIRECT USER SYNC - BYPASSING BROKEN WEBHOOK")
print("=" * 60)

# Import DynamoDB models
from app.adapters.dynamodb import UserProfile

# Connect to PostgreSQL (backend database)
BACKEND_DB_URL = "postgresql://postgres:postgres@localhost:5433/reciprocity_db"

print(f"\n[1] Connecting to PostgreSQL...")
try:
    conn = psycopg2.connect(BACKEND_DB_URL)
    cur = conn.cursor()
    print("  [OK] Connected")
except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)

# Fetch users with completed onboarding
print("\n[2] Fetching completed users...")
cur.execute("""
    SELECT u.id, u.email, u.first_name, u.last_name
    FROM users u
    JOIN roles r ON u.role_id = r.id
    WHERE u.onboarding_status = 'completed'
      AND r.title = 'user'
    ORDER BY u.created_at DESC
""")
users = cur.fetchall()
print(f"  [OK] Found {len(users)} completed users")

if not users:
    print("\nNo users to process. Exiting.")
    cur.close()
    conn.close()
    sys.exit(0)

# Process each user
print("\n[3] Creating DynamoDB profiles...")

processed = 0
skipped = 0
failed = 0

for i, (user_id, email, first_name, last_name) in enumerate(users, 1):
    name = f"{first_name or ''} {last_name or ''}".strip() or email

    # Check if already exists in DynamoDB
    try:
        existing = UserProfile.get(user_id)
        if existing.persona_status == 'completed':
            print(f"  [{i}/{len(users)}] {name}: Already has completed persona")
            skipped += 1
            continue
    except UserProfile.DoesNotExist:
        pass
    except Exception as e:
        print(f"  [{i}/{len(users)}] {name}: Check error - {e}")

    # Get user's onboarding answers
    cur.execute("""
        SELECT prompt, user_response
        FROM user_onboarding_answers
        WHERE user_id = %s
        ORDER BY display_order
    """, (user_id,))
    answers = cur.fetchall()

    if not answers:
        print(f"  [{i}/{len(users)}] {name}: No onboarding answers, skipping")
        skipped += 1
        continue

    # Extract resume link if exists
    resume_link = None
    questions = []

    for prompt, response in answers:
        if not prompt or not response:
            continue

        # Check if response contains resume link (JSON)
        if response and response.startswith('{'):
            try:
                import json
                parsed = json.loads(response)
                if 'resume' in parsed:
                    resume_link = parsed['resume'].get('url') or parsed['resume']
            except:
                pass

        questions.append({
            'prompt': prompt,
            'answer': response
        })

    if not questions:
        print(f"  [{i}/{len(users)}] {name}: No valid questions, skipping")
        skipped += 1
        continue

    # Create user profile in DynamoDB
    try:
        profile = UserProfile.create_user(user_id, resume_link, questions)
        profile.save()
        print(f"  [{i}/{len(users)}] {name}: Created profile ({len(questions)} Q&As)")
        processed += 1
    except Exception as e:
        print(f"  [{i}/{len(users)}] {name}: Create error - {e}")
        failed += 1

cur.close()
conn.close()

print("\n" + "=" * 60)
print(f"SUMMARY: {processed} created, {skipped} skipped, {failed} failed")
print("=" * 60)

# Now trigger persona generation for all profiles without personas
print("\n[4] Triggering persona generation...")
print("  NOTE: Make sure Celery worker is running:")
print("  .venv\\Scripts\\celery -A app.core.celery worker --pool=solo -l info")
print("\n  To generate personas, run:")
print("  .venv\\Scripts\\python.exe -c \"from app.workers.persona_processing import generate_persona_task; [generate_persona_task.delay(p.user_id) for p in __import__('app.adapters.dynamodb', fromlist=['UserProfile']).UserProfile.scan() if p.persona_status != 'completed']\"")
