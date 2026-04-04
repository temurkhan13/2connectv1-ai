"""
DISABLED — Used by criteria_matching_service.py which is superseded by llm_matching_service.py
TODO: Delete when llm_matching_service is confirmed stable.

Test script: Compare criteria-based matching vs old rule-based matching.

Tests specific users that were broken in the old system:
- Troy Dyches (marketing consultant — got other consultants, not clients)
- Kwame Asante (data consultancy — got job seekers, not clients)
- Marcus Chen (fractional CTO — got other CTOs, not startups)
- Jack Jones (founder wanting investors — got other founders)
- ayrah khan (mentor — got founders seeking investors, not mentees)

Also verifies working matches don't regress:
- Shaf Taj (founder → should find investors)
- david park (job seeker → should find hiring CTOs)
- Bob Investor (investor → should find founders)

Usage:
    python scripts/test_criteria_matching.py
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env.production'))

logging.basicConfig(level=logging.WARNING)  # Suppress noisy logs during test
logger = logging.getLogger(__name__)


def main():
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from app.services.criteria_matching_service import find_matches
    from app.adapters.supabase_profiles import UserProfile

    db_url = os.getenv('DATABASE_URL')
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Find test users by name
    test_names = [
        # BROKEN in old system
        "Troy Dyches",
        "Kwame Asante",
        "Marcus Chen",
        "Jack Jones",
        "ayrah khan",
        # WORKING in old system (verify no regression)
        "Shaf taj",
        "david park",
        "Bob Investor",
    ]

    for name in test_names:
        cursor.execute(
            "SELECT user_id, persona_name, persona_designation, "
            "SUBSTRING(persona_requirements, 1, 200) as wants "
            "FROM user_profiles WHERE persona_name ILIKE %s OR persona_designation ILIKE %s LIMIT 1",
            (f'%{name}%', f'%{name}%')
        )
        row = cursor.fetchone()
        if not row:
            print(f"\n{'='*70}")
            print(f"NOT FOUND: {name}")
            continue

        user_id = row['user_id']
        print(f"\n{'='*70}")
        print(f"USER: {row['persona_name']} | {row['persona_designation']}")
        print(f"WANTS: {row['wants']}...")

        # Check if ideal match profile exists
        cursor.execute(
            "SELECT ideal_match_profile FROM user_profiles WHERE user_id = %s",
            (user_id,)
        )
        imp_row = cursor.fetchone()
        if imp_row and imp_row['ideal_match_profile']:
            imp_text = imp_row['ideal_match_profile'][:200]
            print(f"IDEAL MATCH: {imp_text}...")
        else:
            print(f"IDEAL MATCH: NOT GENERATED YET")
            continue

        # Check if ideal_match embedding exists
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM user_embeddings WHERE user_id = %s AND embedding_type = 'ideal_match'",
            (user_id,)
        )
        emb_count = cursor.fetchone()['cnt']
        if emb_count == 0:
            print(f"IDEAL MATCH EMBEDDING: NOT STORED YET")
            continue

        # Run criteria matching
        matches = find_matches(user_id, limit=5)

        if not matches:
            print(f"MATCHES: None found")
            continue

        print(f"\nTOP {len(matches)} CRITERIA MATCHES:")
        for i, m in enumerate(matches):
            cursor.execute(
                "SELECT persona_name, persona_designation, "
                "SUBSTRING(persona_requirements, 1, 150) as wants, "
                "SUBSTRING(persona_offerings, 1, 150) as offers "
                "FROM user_profiles WHERE user_id = %s",
                (m.user_id,)
            )
            mp = cursor.fetchone()
            if not mp:
                continue

            print(f"\n  #{i+1} {m.combined_score:.0%} (fwd:{m.forward_score:.2f} rev:{m.reverse_score:.2f})")
            print(f"  {mp['persona_name']} | {mp['persona_designation']}")
            print(f"  They want: {mp['wants']}...")
            print(f"  They offer: {mp['offers']}...")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
