#!/usr/bin/env python3
"""
Backfill Embeddings Script

Generates embeddings for all existing users who don't have them.
This includes:
1. Basic 2-vector embeddings (requirements, offerings)
2. Multi-vector embeddings (6 dimensions x 2 directions = 12 embeddings)

Run from the reciprocity-ai directory:
    python scripts/backfill_embeddings.py
"""
import os
import sys
import logging
from datetime import datetime

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.adapters.dynamodb import UserProfile
from app.adapters.postgresql import postgresql_adapter
from app.services.embedding_service import embedding_service
from app.services.multi_vector_matcher import multi_vector_matcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_users():
    """Get all users from DynamoDB."""
    try:
        users = list(UserProfile.scan())
        logger.info(f"Found {len(users)} users in DynamoDB")
        return users
    except Exception as e:
        logger.error(f"Error scanning users: {e}")
        return []


def get_user_embedding_count(user_id: str) -> int:
    """Get count of embeddings for a user."""
    try:
        embeddings = postgresql_adapter.get_user_embeddings(user_id)
        return len(embeddings)
    except Exception as e:
        logger.error(f"Error getting embeddings for {user_id}: {e}")
        return 0


def generate_basic_embeddings(user_id: str, persona: dict) -> dict:
    """Generate basic 2-vector embeddings (requirements, offerings)."""
    results = {'requirements': False, 'offerings': False}

    requirements = persona.get('requirements') or persona.get('what_theyre_looking_for') or ''
    offerings = persona.get('offerings') or ''

    if requirements:
        success = embedding_service.store_user_embeddings(
            user_id=user_id,
            requirements=requirements,
            offerings=offerings
        )
        if success:
            results['requirements'] = True
            results['offerings'] = bool(offerings)

    return results


def generate_multi_vector_embeddings(user_id: str, persona: dict) -> dict:
    """Generate multi-vector embeddings (6 dimensions x 2 directions)."""
    results = {'requirements': {}, 'offerings': {}}

    # Build persona_data dict with all relevant fields
    persona_data = {
        # Primary fields from persona
        "primary_goal": persona.get('archetype') or persona.get('user_type') or '',
        "industry": persona.get('industry') or persona.get('focus') or '',
        "stage": persona.get('designation') or '',
        "geography": '',  # Often not collected
        "engagement_style": persona.get('engagement_style') or '',
        "dealbreakers": '',  # Often not collected
        # Full text fields
        "requirements": persona.get('requirements') or persona.get('what_theyre_looking_for') or '',
        "offerings": persona.get('offerings') or '',
    }

    # If primary_goal is empty, try to infer from other fields
    if not persona_data["primary_goal"]:
        profile_essence = persona.get('profile_essence') or ''
        if 'investor' in profile_essence.lower():
            persona_data["primary_goal"] = "investor"
        elif 'founder' in profile_essence.lower():
            persona_data["primary_goal"] = "founder"
        elif 'advisor' in profile_essence.lower():
            persona_data["primary_goal"] = "advisor"

    # If industry is empty, try to extract from profile_essence
    if not persona_data["industry"]:
        profile_essence = persona.get('profile_essence') or ''
        # Common industry keywords
        industries = ['technology', 'healthcare', 'fintech', 'saas', 'ai', 'biotech', 'retail', 'ecommerce']
        for ind in industries:
            if ind in profile_essence.lower():
                persona_data["industry"] = ind
                break

    logger.debug(f"Persona data for {user_id}: {persona_data}")

    # Store requirements embeddings
    req_results = multi_vector_matcher.store_multi_vector_embeddings(
        user_id=user_id,
        persona_data=persona_data,
        direction="requirements"
    )
    results['requirements'] = req_results

    # Store offerings embeddings
    off_results = multi_vector_matcher.store_multi_vector_embeddings(
        user_id=user_id,
        persona_data=persona_data,
        direction="offerings"
    )
    results['offerings'] = off_results

    return results


def update_needs_matchmaking(user: UserProfile) -> bool:
    """Set needs_matchmaking to 'true' so user can be matched."""
    try:
        user.needs_matchmaking = 'true'
        user.save()
        return True
    except Exception as e:
        logger.error(f"Error updating needs_matchmaking for {user.user_id}: {e}")
        return False


def backfill_user(user: UserProfile, force: bool = False) -> dict:
    """
    Backfill embeddings for a single user.

    Args:
        user: UserProfile instance
        force: If True, regenerate even if embeddings exist

    Returns:
        Dict with results
    """
    user_id = user.user_id
    result = {
        'user_id': user_id,
        'had_persona': False,
        'had_embeddings': False,
        'basic_embeddings': {},
        'multi_vector_embeddings': {},
        'needs_matchmaking_updated': False,
        'skipped': False,
        'error': None
    }

    try:
        # Check if user has persona
        if not user.persona or not user.persona.requirements:
            logger.info(f"Skipping {user_id}: No persona or requirements")
            result['skipped'] = True
            return result

        result['had_persona'] = True

        # Check existing embeddings
        existing_count = get_user_embedding_count(user_id)
        result['had_embeddings'] = existing_count > 0

        if existing_count > 0 and not force:
            logger.info(f"Skipping {user_id}: Already has {existing_count} embeddings (use --force to regenerate)")
            result['skipped'] = True
            return result

        # Convert persona to dict
        persona_dict = {
            'name': user.persona.name,
            'archetype': user.persona.archetype,
            'designation': user.persona.designation,
            'experience': user.persona.experience,
            'focus': user.persona.focus,
            'profile_essence': user.persona.profile_essence,
            'investment_philosophy': user.persona.investment_philosophy,
            'strategy': user.persona.strategy,
            'what_theyre_looking_for': user.persona.what_theyre_looking_for,
            'engagement_style': user.persona.engagement_style,
            'requirements': user.persona.requirements,
            'offerings': user.persona.offerings,
            'user_type': user.persona.user_type,
            'industry': user.persona.industry,
        }

        # Generate basic embeddings
        logger.info(f"Generating basic embeddings for {user_id}")
        basic_results = generate_basic_embeddings(user_id, persona_dict)
        result['basic_embeddings'] = basic_results

        # Generate multi-vector embeddings
        logger.info(f"Generating multi-vector embeddings for {user_id}")
        multi_results = generate_multi_vector_embeddings(user_id, persona_dict)
        result['multi_vector_embeddings'] = multi_results

        # Update needs_matchmaking flag
        result['needs_matchmaking_updated'] = update_needs_matchmaking(user)

        # Count total embeddings generated
        total_generated = sum(1 for v in basic_results.values() if v)
        total_generated += sum(1 for v in multi_results.get('requirements', {}).values() if v)
        total_generated += sum(1 for v in multi_results.get('offerings', {}).values() if v)

        logger.info(f"Generated {total_generated} embeddings for {user_id}")

    except Exception as e:
        logger.error(f"Error backfilling {user_id}: {e}")
        result['error'] = str(e)

    return result


def main(force: bool = False, limit: int = None):
    """
    Main backfill function.

    Args:
        force: Regenerate embeddings even if they exist
        limit: Max users to process (for testing)
    """
    logger.info("=" * 60)
    logger.info("EMBEDDING BACKFILL SCRIPT")
    logger.info("=" * 60)

    # Get all users
    users = get_all_users()

    if limit:
        users = users[:limit]
        logger.info(f"Limited to {limit} users")

    # Track statistics
    stats = {
        'total': len(users),
        'processed': 0,
        'skipped': 0,
        'with_persona': 0,
        'had_embeddings': 0,
        'embeddings_generated': 0,
        'errors': 0
    }

    # Process each user
    for i, user in enumerate(users, 1):
        logger.info(f"\n[{i}/{len(users)}] Processing user: {user.user_id}")

        result = backfill_user(user, force=force)

        if result['skipped']:
            stats['skipped'] += 1
        else:
            stats['processed'] += 1

        if result['had_persona']:
            stats['with_persona'] += 1

        if result['had_embeddings']:
            stats['had_embeddings'] += 1

        if result['error']:
            stats['errors'] += 1
        else:
            # Count embeddings generated
            basic = result.get('basic_embeddings', {})
            multi = result.get('multi_vector_embeddings', {})
            count = sum(1 for v in basic.values() if v)
            count += sum(1 for v in multi.get('requirements', {}).values() if v)
            count += sum(1 for v in multi.get('offerings', {}).values() if v)
            stats['embeddings_generated'] += count

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total users:           {stats['total']}")
    logger.info(f"Users with persona:    {stats['with_persona']}")
    logger.info(f"Had existing embeddings: {stats['had_embeddings']}")
    logger.info(f"Processed:             {stats['processed']}")
    logger.info(f"Skipped:               {stats['skipped']}")
    logger.info(f"Embeddings generated:  {stats['embeddings_generated']}")
    logger.info(f"Errors:                {stats['errors']}")

    # Verify by checking PostgreSQL stats
    try:
        pg_stats = postgresql_adapter.get_embedding_stats()
        logger.info(f"\nPostgreSQL embedding stats:")
        logger.info(f"  Total users:       {pg_stats.get('total_users', 'N/A')}")
        logger.info(f"  Users w/embeddings: {pg_stats.get('users_with_embeddings', 'N/A')}")
        logger.info(f"  Total embeddings:  {pg_stats.get('total_embeddings', 'N/A')}")
    except Exception as e:
        logger.error(f"Could not get PostgreSQL stats: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Backfill embeddings for all users')
    parser.add_argument('--force', action='store_true',
                       help='Regenerate embeddings even if they exist')
    parser.add_argument('--limit', type=int, default=None,
                       help='Max users to process (for testing)')
    parser.add_argument('--user', type=str, default=None,
                       help='Process specific user ID only')

    args = parser.parse_args()

    if args.user:
        # Process single user
        try:
            user = UserProfile.get(args.user)
            result = backfill_user(user, force=args.force)
            logger.info(f"Result: {result}")
        except UserProfile.DoesNotExist:
            logger.error(f"User {args.user} not found")
    else:
        main(force=args.force, limit=args.limit)
