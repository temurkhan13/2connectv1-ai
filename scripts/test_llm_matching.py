"""
Test LLM-scored matching on broken + working users.

Usage:
    cd 2connectv1-ai
    PYTHONPATH=. python scripts/test_llm_matching.py
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env.production'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# Suppress noisy logs
logging.getLogger('app.adapters').setLevel(logging.WARNING)
logging.getLogger('app.services.embedding').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def main():
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from app.services.llm_matching_service import find_matches

    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    test_names = [
        # BROKEN in old system
        ("Troy Dyches", "wants e-commerce CLIENTS needing marketing help"),
        ("Kwame Asante", "wants Series B+ startups needing data engineering"),
        ("Marcus Chen", "wants seed/Series A FinTech founders as clients"),
        ("Jack Jones", "wants pre-seed/seed INVESTORS"),
        ("ayrah khan", "wants early-stage founders seeking MENTORSHIP"),
        ("Roberto Vega", "wants LatAm FinTech FOUNDERS"),
        ("Payments Infrastructure", "wants $25M+ Series B INVESTORS"),
        ("HealthTech Founder Seeking Fractional", "wants a fractional CFO"),
        # WORKING in old system (verify no regression)
        ("Victoria Chase", "wants seed-stage FinTech/healthcare FOUNDERS"),
        ("david park", "wants hiring CTOs/companies in FinTech/AI"),
        ("Bob Investor", "wants AI/healthcare FOUNDERS"),
        ("Shaf taj", "wants UK/Europe seed FinTech INVESTORS"),
        ("dean wheeler", "wants digital health Series A INVESTORS"),
        ("Joe Gordon", "wants senior ENGINEERS for payments"),
    ]

    for name, expected in test_names:
        cursor.execute(
            "SELECT user_id, persona_name, persona_designation "
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
        print(f"{row['persona_name']} ({row['persona_designation']})")
        print(f"SHOULD FIND: {expected}")

        matches = find_matches(user_id, limit=5, cosine_limit=50)

        if not matches:
            print("MATCHES: None")
            continue

        print(f"TOP {len(matches)} LLM-SCORED MATCHES:")
        for i, m in enumerate(matches):
            cursor.execute(
                "SELECT persona_name, persona_designation FROM user_profiles WHERE user_id = %s",
                (m.user_id,)
            )
            mp = cursor.fetchone()
            mname = mp['persona_name'] if mp else 'Unknown'
            mrole = mp['persona_designation'] if mp else ''

            print(f"  #{i+1} LLM:{m.llm_score} Cosine:{m.cosine_score:.2f} | {mname} ({mrole})")
            print(f"       Reason: {m.reason}")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
