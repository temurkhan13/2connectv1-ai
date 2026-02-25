#!/usr/bin/env python3
"""
Complete User Pipeline - Generate Personas, Embeddings, and Match Users
Runs synchronously (no Celery required) to complete the full user journey.
"""
import os
import sys
import logging
from datetime import datetime

# Setup before imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env and OVERRIDE system env vars (system has stale OPENAI_API_KEY)
from dotenv import dotenv_values
env_values = dotenv_values(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
for key, value in env_values.items():
    if value:  # Only set if value exists
        os.environ[key] = value

# Fix DynamoDB env vars
os.environ.setdefault('AWS_DEFAULT_REGION', os.getenv('AWS_REGION', 'us-east-1'))
os.environ.setdefault('DYNAMODB_ENDPOINT_URL', os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import app modules after env setup
from app.adapters.dynamodb import UserProfile
from app.services.persona_service import PersonaService
from app.services.matching_service import MatchingService

# Test users
TEST_USERS = {
    "alice": "11111111-1111-1111-1111-111111111111",
    "bob": "22222222-2222-2222-2222-222222222222",
    "charlie": "33333333-3333-3333-3333-333333333333",
    "diana": "44444444-4444-4444-4444-444444444444",
    "eve": "55555555-5555-5555-5555-555555555555",
}

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print('='*70)

def generate_persona_for_user(user_id: str, name: str) -> bool:
    """Generate persona for a single user."""
    try:
        # Get user profile
        user_profile = UserProfile.get(user_id)

        # Check if already completed
        if user_profile.persona_status == 'completed':
            print(f"    [SKIP] {name}: Persona already completed")
            return True

        # Extract questions
        questions = [q.as_dict() for q in user_profile.profile.raw_questions]
        resume_text = user_profile.resume_text.text if user_profile.resume_text and user_profile.resume_text.text else ""

        if not questions:
            print(f"    [FAIL] {name}: No questions found")
            return False

        # Generate persona
        persona_service = PersonaService()
        persona_data = persona_service.generate_persona_sync(questions, resume_text)

        if persona_data:
            persona = persona_data.get('persona', {})
            requirements = persona_data.get('requirements', '')
            offerings = persona_data.get('offerings', '')

            # Store persona
            user_profile.update(
                actions=[
                    UserProfile.persona.name.set(persona.get('name')),
                    UserProfile.persona.archetype.set(persona.get('archetype')),
                    UserProfile.persona.experience.set(persona.get('experience')),
                    UserProfile.persona.focus.set(persona.get('focus')),
                    UserProfile.persona.profile_essence.set(persona.get('profile_essence')),
                    UserProfile.persona.investment_philosophy.set(persona.get('investment_philosophy')),
                    UserProfile.persona.what_theyre_looking_for.set(persona.get('what_theyre_looking_for')),
                    UserProfile.persona.engagement_style.set(persona.get('engagement_style')),
                    UserProfile.persona.designation.set(persona.get('designation')),
                    UserProfile.persona.requirements.set(requirements),
                    UserProfile.persona.offerings.set(offerings),
                    UserProfile.persona.generated_at.set(datetime.utcnow()),
                    UserProfile.persona_status.set('completed')
                ]
            )

            print(f"    [PASS] {name}: {persona.get('name', 'Unknown')} - {persona.get('archetype', 'N/A')}")
            return True
        else:
            print(f"    [FAIL] {name}: Persona generation returned None")
            return False

    except Exception as e:
        print(f"    [FAIL] {name}: {str(e)[:60]}")
        return False

def generate_embeddings_for_user(user_id: str, name: str) -> bool:
    """Generate embeddings for a single user."""
    try:
        from app.services.embedding_service import EmbeddingService

        # Get user profile
        user_profile = UserProfile.get(user_id)

        if user_profile.persona_status != 'completed':
            print(f"    [SKIP] {name}: Persona not completed")
            return False

        # Get persona data
        persona = user_profile.persona
        requirements = persona.requirements or ""
        offerings = persona.offerings or ""

        if not requirements and not offerings:
            print(f"    [FAIL] {name}: No requirements or offerings in persona")
            return False

        # Use the built-in store_user_embeddings method
        embedding_service = EmbeddingService()
        success = embedding_service.store_user_embeddings(user_id, requirements, offerings)

        if success:
            print(f"    [PASS] {name}: Embeddings stored for requirements + offerings")
            return True
        else:
            print(f"    [FAIL] {name}: Failed to store embeddings")
            return False

    except ImportError as e:
        print(f"    [SKIP] {name}: Embedding service not available ({e})")
        return False
    except Exception as e:
        print(f"    [FAIL] {name}: {str(e)[:60]}")
        return False

def find_matches_for_user(user_id: str, name: str) -> dict:
    """Find matches for a user."""
    try:
        matching_service = MatchingService()
        result = matching_service.find_user_matches(user_id)

        # Combine requirements and offerings matches
        req_matches = result.get('requirements_matches', [])
        off_matches = result.get('offerings_matches', [])

        # Get unique match user IDs
        all_match_ids = set()
        for m in req_matches:
            all_match_ids.add(m.get('user_id'))
        for m in off_matches:
            all_match_ids.add(m.get('user_id'))

        if all_match_ids:
            match_names = []
            for match_id in all_match_ids:
                # Find name by ID
                for n, uid in TEST_USERS.items():
                    if uid == match_id:
                        match_names.append(n)
                        break

            print(f"    [PASS] {name}: {len(all_match_ids)} matches - {', '.join(match_names) if match_names else 'external users'}")
            return {"user": name, "match_count": len(all_match_ids), "matches": match_names, "req_matches": len(req_matches), "off_matches": len(off_matches)}
        else:
            print(f"    [INFO] {name}: No matches found (threshold may be too high)")
            return {"user": name, "match_count": 0, "matches": []}

    except Exception as e:
        error_msg = str(e)
        if "persona" in error_msg.lower() or "not completed" in error_msg.lower():
            print(f"    [SKIP] {name}: Persona not ready for matching")
        else:
            print(f"    [FAIL] {name}: {error_msg[:60]}")
        return {"user": name, "match_count": -1, "matches": [], "error": error_msg[:60]}

def main():
    print("\n" + "="*70)
    print("  RECIPROCITY AI - Complete User Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Phase 1: Generate Personas
    print_header("PHASE 1: GENERATE PERSONAS (via OpenAI)")
    persona_results = {}
    for name, user_id in TEST_USERS.items():
        result = generate_persona_for_user(user_id, name)
        persona_results[name] = result

    # Check persona status
    print("\n  Persona Status After Generation:")
    for name, user_id in TEST_USERS.items():
        try:
            user = UserProfile.get(user_id)
            persona_name = user.persona.name if user.persona else "N/A"
            print(f"    {name:10} status={user.persona_status:15} name={persona_name}")
        except Exception as e:
            print(f"    {name:10} Error: {e}")

    # Phase 2: Generate Embeddings
    print_header("PHASE 2: GENERATE EMBEDDINGS")
    embedding_results = {}
    for name, user_id in TEST_USERS.items():
        if persona_results.get(name):
            result = generate_embeddings_for_user(user_id, name)
            embedding_results[name] = result
        else:
            print(f"    [SKIP] {name}: Skipped (persona failed)")
            embedding_results[name] = False

    # Phase 3: Find Matches
    print_header("PHASE 3: FIND MATCHES")
    match_results = {}
    for name, user_id in TEST_USERS.items():
        if embedding_results.get(name):
            result = find_matches_for_user(user_id, name)
            match_results[name] = result
        else:
            print(f"    [SKIP] {name}: Skipped (no embeddings)")
            match_results[name] = {"user": name, "match_count": -1, "matches": [], "error": "No embeddings"}

    # Summary
    print_header("MATCHING SUMMARY")

    print("\n  Match Matrix:")
    print("  " + "-"*66)
    print(f"  {'User':12} | Matches With")
    print("  " + "-"*66)

    for name, result in match_results.items():
        matches = result.get("matches", [])
        if result.get("match_count", -1) >= 0:
            match_str = ", ".join(matches) if matches else "(no test user matches)"
            print(f"  {name:12} | {match_str}")
        else:
            print(f"  {name:12} | (skipped - {result.get('error', 'unknown error')})")

    print("  " + "-"*66)

    # Final stats
    print_header("FINAL STATISTICS")

    personas_ok = sum(1 for v in persona_results.values() if v)
    embeddings_ok = sum(1 for v in embedding_results.values() if v)
    matches_found = sum(1 for v in match_results.values() if v.get("match_count", 0) > 0)

    print(f"  Personas Generated:  {personas_ok}/5")
    print(f"  Embeddings Created:  {embeddings_ok}/5")
    print(f"  Users With Matches:  {matches_found}/5")

    print("\n" + "="*70)
    if personas_ok == 5 and embeddings_ok == 5:
        print("  STATUS: PIPELINE COMPLETE - ALL USERS PROCESSED")
    else:
        print("  STATUS: PIPELINE INCOMPLETE - CHECK LOGS")
    print("="*70 + "\n")

    return 0 if personas_ok == 5 else 1

if __name__ == "__main__":
    sys.exit(main())
