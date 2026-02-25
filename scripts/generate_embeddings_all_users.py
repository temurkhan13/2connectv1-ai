"""
Generate Embeddings for All Users
=================================
Generates semantic embeddings for all users who have completed personas.

Usage:
    cd reciprocity-ai
    .venv/Scripts/python.exe scripts/generate_embeddings_all_users.py
"""
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

print("=" * 70)
print("GENERATE EMBEDDINGS FOR ALL USERS")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Import services after dotenv
from app.adapters.dynamodb import UserProfile
from app.services.embedding_service import embedding_service
from app.adapters.postgresql import postgresql_adapter

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

# Generate embeddings for each user
print(f"\n[2] Generating embeddings for {len(users)} users...")
print("-" * 70)

generated = 0
skipped = 0
failed = 0

for i, user in enumerate(users, 1):
    user_id = user.user_id
    name = user.persona.name if user.persona else user_id[:30]

    # Check if embeddings already exist
    existing = postgresql_adapter.get_user_embeddings(user_id)
    if existing.get('requirements') and existing.get('offerings'):
        print(f"[{i:3}/{len(users)}] {name[:35]:35} -> Already has embeddings")
        skipped += 1
        continue

    # Check if persona has requirements and offerings
    persona = user.persona
    if not persona:
        print(f"[{i:3}/{len(users)}] {name[:35]:35} -> No persona, skipping")
        skipped += 1
        continue

    requirements = persona.requirements if hasattr(persona, 'requirements') else None
    offerings = persona.offerings if hasattr(persona, 'offerings') else None

    if not requirements and not offerings:
        print(f"[{i:3}/{len(users)}] {name[:35]:35} -> No req/off, skipping")
        skipped += 1
        continue

    try:
        # Generate requirements embedding
        if requirements:
            req_text = " ".join(requirements) if isinstance(requirements, list) else str(requirements)
            req_vector = embedding_service.generate_embedding(req_text)
            if req_vector:
                postgresql_adapter.store_embedding(
                    user_id=user_id,
                    embedding_type='requirements',
                    vector_data=req_vector,
                    metadata={'source': 'persona.requirements'}
                )

        # Generate offerings embedding
        if offerings:
            off_text = " ".join(offerings) if isinstance(offerings, list) else str(offerings)
            off_vector = embedding_service.generate_embedding(off_text)
            if off_vector:
                postgresql_adapter.store_embedding(
                    user_id=user_id,
                    embedding_type='offerings',
                    vector_data=off_vector,
                    metadata={'source': 'persona.offerings'}
                )

        print(f"[{i:3}/{len(users)}] {name[:35]:35} -> Embeddings generated")
        generated += 1

    except Exception as e:
        print(f"[{i:3}/{len(users)}] {name[:35]:35} -> ERROR: {str(e)[:40]}")
        failed += 1

print("-" * 70)
print()
print("=" * 70)
print("EMBEDDING GENERATION SUMMARY")
print("=" * 70)
print(f"Total users:      {len(users)}")
print(f"Generated:        {generated}")
print(f"Already existed:  {skipped}")
print(f"Failed:           {failed}")
print("=" * 70)
