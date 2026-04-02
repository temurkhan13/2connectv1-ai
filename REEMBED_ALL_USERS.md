# Re-Embed All Users: Migration from all-mpnet-base-v2 to Gemini text-embedding-004

## Why This Is Needed

We switched the embedding model from local `all-mpnet-base-v2` (SentenceTransformers) to `Gemini text-embedding-004`. Both produce 768-dimension vectors, but they exist in **completely different vector spaces** — vectors from one model cannot be compared against vectors from the other.

**Current state after deployment:**
- New users → Gemini embeddings (correct)
- Existing users → all-mpnet-base-v2 embeddings (stale)
- Matching between old and new users → **broken similarity scores**

**After running this script:**
- All users → Gemini embeddings
- All matching → consistent vector space
- Matches should be recalculated after re-embedding

---

## Pre-requisites

1. **Env var `USE_GEMINI_EMBEDDINGS=true`** must be set on Render AI service (already done)
2. **Env var `GEMINI_EMBEDDINGS_KEY`** must be set on Render AI service (already done)
3. The AI service must be deployed and healthy
4. You need direct access to the AI service (either SSH or run locally with production env vars)

---

## Option A: Run via API endpoint (Recommended)

The AI service already has admin endpoints. The script below creates a temporary admin endpoint for re-embedding.

### Step 1: Add the re-embedding endpoint

Add this to `app/routers/admin.py` (or whichever admin router exists):

```python
@router.post("/admin/reembed-all-users")
async def reembed_all_users(
    request: Request,
    background_tasks: BackgroundTasks,
    dry_run: bool = Query(default=True, description="If true, only count users without re-embedding"),
    batch_size: int = Query(default=10, description="Users per batch"),
):
    """
    Re-embed all user profiles using the currently configured embedding model.
    This is a one-time migration endpoint for switching embedding models.
    
    MUST be called with admin API key.
    """
    from app.services.reembed_script import run_reembedding
    
    if dry_run:
        result = run_reembedding(dry_run=True)
        return {"status": "dry_run", "result": result}
    
    # Run in background to avoid timeout
    background_tasks.add_task(run_reembedding, dry_run=False, batch_size=batch_size)
    return {"status": "started", "message": "Re-embedding started in background. Check logs for progress."}
```

### Step 2: Create the re-embedding script

Create file `app/services/reembed_script.py`:

```python
"""
One-time migration script: Re-embed all user profiles with current embedding model.

Reads all user profiles from Supabase, generates new embeddings using the
currently configured model (Gemini text-embedding-004), and overwrites
existing embeddings in pgvector.

Usage:
    # Dry run (count only)
    python -m app.services.reembed_script --dry-run
    
    # Full run
    python -m app.services.reembed_script
    
    # With batch size
    python -m app.services.reembed_script --batch-size 20
"""
import os
import sys
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def run_reembedding(dry_run: bool = True, batch_size: int = 10) -> Dict[str, Any]:
    """
    Re-embed all users with the current embedding model.
    
    Args:
        dry_run: If True, only count and report — don't write anything
        batch_size: Number of users to process before sleeping (rate limit protection)
    
    Returns:
        Summary dict with counts
    """
    from app.adapters.supabase_profiles import UserProfile
    from app.services.embedding_service import embedding_service
    from app.services.multi_vector_embedding_service import multi_vector_service
    from app.adapters.postgresql import postgresql_adapter
    
    logger.info(f"=== RE-EMBEDDING ALL USERS ===")
    logger.info(f"Model: {embedding_service.model_name}")
    logger.info(f"Dimension: {embedding_service.embedding_dimension}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"Batch size: {batch_size}")
    
    stats = {
        "total_profiles": 0,
        "profiles_with_persona": 0,
        "profiles_skipped_no_text": 0,
        "embeddings_generated": 0,
        "embeddings_failed": 0,
        "multi_vector_generated": 0,
        "model": embedding_service.model_name,
        "dimension": embedding_service.embedding_dimension,
        "dry_run": dry_run,
    }
    
    # Scan all profiles
    profiles = list(UserProfile.scan())
    stats["total_profiles"] = len(profiles)
    logger.info(f"Found {len(profiles)} user profiles")
    
    batch_count = 0
    
    for i, profile in enumerate(profiles):
        user_id = profile.user_id
        
        # Extract text from persona
        requirements_text = ""
        offerings_text = ""
        user_type = None
        
        if profile.persona:
            p = profile.persona
            requirements_text = getattr(p, 'requirements', '') or ''
            offerings_text = getattr(p, 'offerings', '') or ''
            user_type = getattr(p, 'archetype', None) or getattr(p, 'primary_goal', None)
            
            # If requirements/offerings are empty, try to build from other fields
            if not requirements_text:
                parts = []
                if getattr(p, 'focus', ''): parts.append(f"Focus: {p.focus}")
                if getattr(p, 'designation', ''): parts.append(f"Role: {p.designation}")
                requirements_text = ". ".join(parts)
            
            if not offerings_text:
                parts = []
                if getattr(p, 'expertise', ''): parts.append(f"Expertise: {p.expertise}")
                if getattr(p, 'designation', ''): parts.append(f"Background: {p.designation}")
                offerings_text = ". ".join(parts)
        
        if not requirements_text and not offerings_text:
            stats["profiles_skipped_no_text"] += 1
            logger.debug(f"[{i+1}/{len(profiles)}] Skipping {user_id[:8]} — no text to embed")
            continue
        
        stats["profiles_with_persona"] += 1
        
        if dry_run:
            logger.info(f"[{i+1}/{len(profiles)}] Would re-embed {user_id[:8]} "
                       f"(req={len(requirements_text)} chars, off={len(offerings_text)} chars, type={user_type})")
            continue
        
        # === LIVE: Generate and store new embeddings ===
        try:
            # 1. Basic embeddings (requirements + offerings)
            success = embedding_service.store_user_embeddings(
                user_id=user_id,
                requirements=requirements_text,
                offerings=offerings_text
            )
            
            if success:
                stats["embeddings_generated"] += 1
            else:
                stats["embeddings_failed"] += 1
                logger.warning(f"[{i+1}/{len(profiles)}] Basic embedding failed for {user_id[:8]}")
                continue
            
            # 2. Multi-vector dimension embeddings
            try:
                mv_result = multi_vector_service.generate_multi_vector_embeddings(
                    user_id=user_id,
                    requirements_text=requirements_text,
                    offerings_text=offerings_text,
                    store_in_db=True,
                    user_type=user_type
                )
                dim_count = len(mv_result.get("dimensions", {}))
                stats["multi_vector_generated"] += dim_count
                logger.info(f"[{i+1}/{len(profiles)}] Re-embedded {user_id[:8]} — "
                           f"basic=OK, multi_vector={dim_count} dims, type={user_type}")
            except Exception as mv_err:
                logger.warning(f"[{i+1}/{len(profiles)}] Multi-vector failed for {user_id[:8]}: {mv_err}")
                # Basic embedding succeeded, that's enough
            
            # Rate limit protection: pause between batches
            batch_count += 1
            if batch_count >= batch_size:
                batch_count = 0
                logger.info(f"Batch complete, sleeping 2s for rate limit protection...")
                time.sleep(2)
                
        except Exception as e:
            stats["embeddings_failed"] += 1
            logger.error(f"[{i+1}/{len(profiles)}] Failed to re-embed {user_id[:8]}: {e}")
    
    # Summary
    logger.info(f"=== RE-EMBEDDING COMPLETE ===")
    logger.info(f"Total profiles: {stats['total_profiles']}")
    logger.info(f"With persona text: {stats['profiles_with_persona']}")
    logger.info(f"Skipped (no text): {stats['profiles_skipped_no_text']}")
    logger.info(f"Embeddings generated: {stats['embeddings_generated']}")
    logger.info(f"Embeddings failed: {stats['embeddings_failed']}")
    logger.info(f"Multi-vector dims: {stats['multi_vector_generated']}")
    
    return stats


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Re-embed all user profiles")
    parser.add_argument("--dry-run", action="store_true", default=False,
                       help="Count profiles without generating embeddings")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Users per batch before rate limit pause")
    args = parser.parse_args()
    
    result = run_reembedding(dry_run=args.dry_run, batch_size=args.batch_size)
    print(f"\n{'='*50}")
    print(f"Result: {result}")
```

### Step 3: Run it

**Option 1: Via Render SSH (recommended for production)**

```bash
# SSH into the AI service
ssh srv-d6fclni4d50c73eaa7fg@ssh.oregon.render.com

# Dry run first — see how many users will be affected
cd /opt/render/project/src
python -m app.services.reembed_script --dry-run

# If dry run looks good, run for real
python -m app.services.reembed_script --batch-size 10
```

**Option 2: Via API endpoint (if you added the admin endpoint)**

```bash
# Dry run
curl -X POST "https://twoconnectv1-ai.onrender.com/api/v1/admin/reembed-all-users?dry_run=true" \
  -H "X-API-Key: YOUR_ADMIN_KEY"

# Live run (runs in background)
curl -X POST "https://twoconnectv1-ai.onrender.com/api/v1/admin/reembed-all-users?dry_run=false&batch_size=10" \
  -H "X-API-Key: YOUR_ADMIN_KEY"
```

**Option 3: Run locally with production env vars**

```bash
cd c:\Users\hp\AI_Agent_Workspace\2connectv1-ai

# Copy production env vars from Render dashboard to a local .env.production file
# MUST include: DATABASE_URL, REDIS_URL, GEMINI_EMBEDDINGS_KEY, USE_GEMINI_EMBEDDINGS=true

# Load production env and run
set DOTENV_FILE=.env.production
python -m app.services.reembed_script --dry-run
python -m app.services.reembed_script --batch-size 10
```

---

## Step 4: After Re-Embedding — Recalculate Matches

After all users are re-embedded, existing match scores are stale (computed from old embeddings). Trigger a full match recalculation:

```bash
# Via admin dashboard or API
curl -X POST "https://twoconnectv1-ai.onrender.com/api/v1/admin/regenerate-all-matches" \
  -H "X-API-Key: YOUR_ADMIN_KEY"
```

Or from the admin dashboard: **Matching Diagnostics → Regenerate Matches** for each user.

---

## Step 5: Verify

1. Check Render logs for `=== RE-EMBEDDING COMPLETE ===` summary
2. Check admin dashboard → System Health → Embedding stats should show `Gemini API + pgvector`
3. Pick a user from admin dashboard → check their embeddings have `updated_at` timestamp from today
4. Test a new onboarding → verify match quality looks reasonable

---

## Rollback

If something goes wrong:

1. Set `USE_GEMINI_EMBEDDINGS=false` on Render env vars
2. The system falls back to local `all-mpnet-base-v2`
3. Run the re-embed script again to regenerate with the local model
4. Old embeddings are overwritten via `ON CONFLICT (user_id, embedding_type) DO UPDATE`

---

## Expected Numbers

Based on current data:
- ~150 user profiles in system
- ~722 matches
- Re-embedding should take ~5-10 minutes with batch_size=10
- Gemini free tier allows 1500 requests/min — well within limits
- Each user generates ~15+ embeddings (requirements, offerings, + dimension-specific)
- Total API calls: ~150 users × 15 embeddings = ~2,250 calls

---

## Files Created/Modified

| File | Action |
|------|--------|
| `app/services/reembed_script.py` | **CREATE** — the migration script |
| `app/routers/admin.py` | **MODIFY** — add `/admin/reembed-all-users` endpoint (optional) |
| No other files need changing | Embedding service already configured for Gemini |
