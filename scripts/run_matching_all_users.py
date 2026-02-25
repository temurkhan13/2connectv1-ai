"""
Run Matching Algorithm for All Users
=====================================
Finds and stores matches for all users who have completed personas.

Usage:
    cd reciprocity-ai
    .venv/Scripts/python.exe scripts/run_matching_all_users.py
"""
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

print("=" * 70)
print("RUN MATCHING ALGORITHM FOR ALL USERS")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Import services after dotenv
from app.adapters.dynamodb import UserProfile, UserMatches
from app.services.matching_service import matching_service

# Get all users with completed personas
print("[1] Fetching users with completed personas...")
try:
    users = list(UserProfile.scan(UserProfile.persona_status == "completed"))
    print(f"    Found {len(users)} users with completed personas")
except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

if not users:
    print("\nNo users to process. Exiting.")
    sys.exit(0)

# Process each user
print(f"\n[2] Running matching for {len(users)} users...")
print("-" * 70)

processed = 0
with_matches = 0
no_matches = 0
failed = 0
total_matches_found = 0

for i, user in enumerate(users, 1):
    user_id = user.user_id
    name = user.persona.name if user.persona else user_id[:30]

    try:
        # Find and store matches
        result = matching_service.find_and_store_user_matches(user_id)

        if result.get('success'):
            total = result.get('total_matches', 0)
            req_count = result.get('requirements_count', 0)
            off_count = result.get('offerings_count', 0)

            if total > 0:
                print(f"[{i:3}/{len(users)}] {name[:35]:35} -> {total} matches (req:{req_count}, off:{off_count})")
                with_matches += 1
                total_matches_found += total
            else:
                print(f"[{i:3}/{len(users)}] {name[:35]:35} -> 0 matches")
                no_matches += 1
            processed += 1
        else:
            print(f"[{i:3}/{len(users)}] {name[:35]:35} -> FAILED: {result.get('message', 'Unknown error')}")
            failed += 1

    except Exception as e:
        print(f"[{i:3}/{len(users)}] {user_id[:35]:35} -> ERROR: {str(e)[:40]}")
        failed += 1

print("-" * 70)
print()
print("=" * 70)
print("MATCHING SUMMARY")
print("=" * 70)
print(f"Total users processed:     {processed}")
print(f"Users with matches:        {with_matches}")
print(f"Users without matches:     {no_matches}")
print(f"Failed:                    {failed}")
print(f"Total match pairs found:   {total_matches_found}")
print("=" * 70)

# Print sample matches for verification
if with_matches > 0:
    print("\n[3] Sample matches (first 3 users with matches):")
    print("-" * 70)

    sample_count = 0
    for user in users:
        if sample_count >= 3:
            break

        try:
            stored = UserMatches.get_user_matches(user.user_id)
            if stored and stored.get('total_matches', 0) > 0:
                name = user.persona.name if user.persona else user.user_id[:30]
                print(f"\n{name}:")

                req_matches = stored.get('requirements_matches', [])[:2]
                for m in req_matches:
                    score = m.get('similarity_score', 0)
                    target = m.get('user_id', '?')[:30]
                    print(f"  -> Needs match: {target} (score: {score:.2f})")

                sample_count += 1
        except:
            pass

    print()
