"""
Standalone re-embedding script for all 81 users.

Upgraded from text-embedding-004 (768 dims, DEPRECATED) to:
  Model: gemini-embedding-2-preview (8K token input, multimodal)
  Dimensions: 1536 (same quality as 3072, compatible with pgvector HNSW index)

PREREQUISITES:
  1. SQL migration already ran (vector column = VECTOR(1536), old embeddings truncated)
  2. Render env vars updated (GEMINI_EMBEDDING_MODEL, EMBEDDING_DIMENSION)

USAGE:
  cd 2connectv1-ai
  python scripts/reembed_all_users.py --dry-run     # Check what will happen
  python scripts/reembed_all_users.py                # Run for real

HOW IT WORKS:
  1. Reads all user profiles from Supabase (user_profiles table)
  2. For each user with persona data, takes requirements + offerings text
  3. Generates new 1536-dim embeddings using gemini-embedding-2-preview
  4. Stores in user_embeddings table (pgvector)
  5. Rate-limited: 10 users per batch, 2s pause between batches

NOTE: This re-embeds using EXISTING persona text (requirements + offerings).
New users who onboard after the code deploy will get ENRICHED personas
(from full conversation text), which will produce even better embeddings.
"""
import os
import sys
import time
import logging
import argparse
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load env from .env.production for local runs
env_file = os.path.join(os.path.dirname(__file__), '..', '.env.production')
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)
    print(f"Loaded env from: {env_file}")
else:
    load_dotenv()
    print("Using default .env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Configuration ───
DATABASE_URL = os.getenv('DATABASE_URL')
GEMINI_KEY = os.getenv('GEMINI_EMBEDDINGS_KEY')
MODEL_NAME = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/gemini-embedding-2-preview')
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIMENSION', '1536'))

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Load .env.production or set it manually.")
    sys.exit(1)
if not GEMINI_KEY:
    print("ERROR: GEMINI_EMBEDDINGS_KEY not set.")
    sys.exit(1)


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)


def init_gemini():
    """Initialize Gemini embedding client."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)
    return genai


def generate_embedding(genai_client, text: str) -> list:
    """Generate embedding using gemini-embedding-2-preview.

    Note: embedding-2 does NOT support task_type parameter.
    """
    if not text or not text.strip():
        return None

    try:
        result = genai_client.embed_content(
            model=MODEL_NAME,
            content=text.strip(),
        )
        embedding = result['embedding']

        # Truncate to target dimension if model returns more
        if len(embedding) > EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]

        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def store_embedding(conn, user_id: str, embedding_type: str, vector_data: list):
    """Store embedding in pgvector."""
    try:
        # Register pgvector
        import pgvector.psycopg2
        pgvector.psycopg2.register_vector(conn)
    except Exception:
        pass  # Already registered

    import numpy as np
    vector = np.array(vector_data, dtype=np.float32)

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_embeddings (user_id, embedding_type, vector_data, metadata, created_at, updated_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, embedding_type) DO UPDATE SET
            vector_data = EXCLUDED.vector_data,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP
    """, (
        user_id,
        embedding_type,
        vector,
        '{"model": "gemini-embedding-2-preview", "dimension": ' + str(EMBEDDING_DIM) + '}',
    ))
    conn.commit()
    cursor.close()


def get_all_profiles(conn):
    """Fetch all user profiles with persona data."""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT user_id, persona_name, persona_requirements, persona_offerings,
               persona_archetype, persona_focus, persona_designation,
               persona_profile_essence
        FROM user_profiles
        WHERE persona_status = 'completed'
        ORDER BY created_at
    """)
    profiles = cursor.fetchall()
    cursor.close()
    return profiles


def build_embedding_text(profile: dict, direction: str) -> str:
    """Build rich text for embedding from profile fields.

    Combines multiple persona fields for richer embedding input.
    """
    parts = []

    if direction == 'requirements':
        # What this user needs
        req = profile.get('persona_requirements') or ''
        if req and req != 'Not specified':
            parts.append(req)
        # Add profile essence for context
        essence = profile.get('persona_profile_essence') or ''
        if essence and essence != 'Not specified':
            parts.append(essence)
    else:
        # What this user offers
        off = profile.get('persona_offerings') or ''
        if off and off != 'Not specified':
            parts.append(off)
        # Add focus and designation for richer context
        focus = profile.get('persona_focus') or ''
        if focus and focus != 'Not specified':
            parts.append(f"Focus areas: {focus}")
        designation = profile.get('persona_designation') or ''
        if designation and designation != 'Not specified':
            parts.append(f"Role: {designation}")
        essence = profile.get('persona_profile_essence') or ''
        if essence and essence != 'Not specified':
            parts.append(essence)

    text = ". ".join(parts)
    return text if text.strip() else ""


def run(dry_run: bool = True, batch_size: int = 10):
    """Main re-embedding loop."""
    print(f"\n{'='*60}")
    print(f"RE-EMBEDDING ALL USERS")
    print(f"{'='*60}")
    print(f"Model:     {MODEL_NAME}")
    print(f"Dimension: {EMBEDDING_DIM}")
    print(f"Mode:      {'DRY RUN' if dry_run else '*** LIVE ***'}")
    print(f"Batch:     {batch_size} users per batch")
    print(f"{'='*60}\n")

    conn = get_db_connection()

    # Check current embedding count
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM user_embeddings")
    existing = cursor.fetchone()[0]
    cursor.close()
    print(f"Existing embeddings in DB: {existing}")

    # Get all profiles
    profiles = get_all_profiles(conn)
    print(f"Profiles with persona: {len(profiles)}\n")

    if not profiles:
        print("No profiles found. Nothing to do.")
        conn.close()
        return

    # Init Gemini
    genai_client = None
    if not dry_run:
        genai_client = init_gemini()
        # Test with a simple embedding
        test = generate_embedding(genai_client, "test embedding")
        if test:
            print(f"Gemini test OK: got {len(test)}-dim vector\n")
        else:
            print("ERROR: Gemini test embedding failed. Check API key.")
            conn.close()
            return

    stats = {
        "total": len(profiles),
        "embedded": 0,
        "skipped": 0,
        "failed": 0,
    }

    batch_count = 0

    for i, profile in enumerate(profiles):
        user_id = str(profile['user_id'])
        name = profile.get('persona_name') or user_id[:8]

        req_text = build_embedding_text(profile, 'requirements')
        off_text = build_embedding_text(profile, 'offerings')

        if not req_text and not off_text:
            stats["skipped"] += 1
            logger.info(f"[{i+1}/{len(profiles)}] SKIP {name} — no text")
            continue

        if dry_run:
            logger.info(f"[{i+1}/{len(profiles)}] WOULD embed {name} "
                       f"(req={len(req_text)} chars, off={len(off_text)} chars)")
            stats["embedded"] += 1
            continue

        # Generate and store
        try:
            success = False

            if req_text:
                req_vec = generate_embedding(genai_client, req_text)
                if req_vec:
                    store_embedding(conn, user_id, 'requirements', req_vec)
                    success = True
                else:
                    logger.warning(f"[{i+1}/{len(profiles)}] Requirements embedding failed for {name}")

            if off_text:
                off_vec = generate_embedding(genai_client, off_text)
                if off_vec:
                    store_embedding(conn, user_id, 'offerings', off_vec)
                    success = True
                else:
                    logger.warning(f"[{i+1}/{len(profiles)}] Offerings embedding failed for {name}")

            if success:
                stats["embedded"] += 1
                logger.info(f"[{i+1}/{len(profiles)}] OK {name} "
                           f"(req={len(req_text)} chars, off={len(off_text)} chars)")
            else:
                stats["failed"] += 1

            # Rate limit: pause between batches
            batch_count += 1
            if batch_count >= batch_size:
                batch_count = 0
                logger.info(f"--- Batch pause (2s) ---")
                time.sleep(2)

        except Exception as e:
            stats["failed"] += 1
            logger.error(f"[{i+1}/{len(profiles)}] FAIL {name}: {e}")

    conn.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total profiles:  {stats['total']}")
    print(f"Embedded:        {stats['embedded']}")
    print(f"Skipped (empty): {stats['skipped']}")
    print(f"Failed:          {stats['failed']}")
    print(f"{'='*60}")

    if dry_run:
        print("\nThis was a DRY RUN. Run without --dry-run to execute.")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed all users with gemini-embedding-2-preview")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't write")
    parser.add_argument("--batch-size", type=int, default=10, help="Users per batch (default: 10)")
    args = parser.parse_args()

    run(dry_run=args.dry_run, batch_size=args.batch_size)
