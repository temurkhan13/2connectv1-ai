"""
Step 1 of embedding upgrade: Re-generate richer AI summaries for all 81 users.

This script reads each user's existing Q&A data and resume text, then re-runs
persona generation using the updated prompt that produces 1,500-2,000 word summaries
(previously ~300-500 words).

After this completes, run reembed_all_users.py (Step 2) to generate new embeddings
from the enriched persona text.

ORDER OF OPERATIONS:
  Step 1: python scripts/enrich_personas.py           ← THIS SCRIPT
  Step 2: python scripts/reembed_all_users.py          ← After Step 1 completes

WHAT IT DOES:
  For each user with a completed persona:
  1. Reads raw_questions (Q&A pairs from onboarding) + resume_text
  2. Calls Sonnet 4.6 with updated prompt requesting 1,500-2,000 word output
  3. Updates persona fields in user_profiles (profile_essence, strategy,
     requirements, offerings, what_looking_for, etc.)
  4. Updates user_summaries with new markdown summary

WHAT IT DOES NOT DO:
  - Does NOT change slot data or onboarding answers
  - Does NOT generate embeddings (that's Step 2)
  - Does NOT fabricate information — only expands on existing data

USAGE:
  cd 2connectv1-ai
  python scripts/enrich_personas.py --dry-run          # Preview
  python scripts/enrich_personas.py --dry-run --user USER_ID  # Preview single user
  python scripts/enrich_personas.py                    # Run all users
  python scripts/enrich_personas.py --user USER_ID     # Run single user
"""
import os
import sys
import json
import time
import logging
import argparse
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load env
env_file = os.path.join(os.path.dirname(__file__), '..', '.env.production')
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)
    print(f"Loaded env from: {env_file}")
else:
    load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Config ───
DATABASE_URL = os.getenv('DATABASE_URL')
ANTHROPIC_KEY = os.getenv('ANTHROPIC_MATCHING_KEY') or os.getenv('ANTHROPIC_API_KEY')
MODEL = "claude-sonnet-4-6"

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set.")
    sys.exit(1)
if not ANTHROPIC_KEY:
    print("ERROR: No Anthropic API key found (ANTHROPIC_MATCHING_KEY or ANTHROPIC_API_KEY).")
    sys.exit(1)

# ─── Persona generation prompt (same as updated persona_prompts.py) ───
ENRICHMENT_PROMPT = """You are an expert persona generation system. You are ENRICHING an existing persona to be more detailed.

You will receive the user's original Q&A responses and optionally their resume. Your job is to generate a MUCH RICHER version of their persona — aim for 1,500-2,000 total words across all fields.

CRITICAL RULES:
- Use ONLY the provided data. Do NOT invent facts, companies, numbers, or achievements.
- If the original data is thin (short answers, no resume), you can ELABORATE on what's there but NEVER fabricate.
- Expand "FinTech" into "financial technology sector" with context from their answers.
- Expand "10 years experience" into a narrative about their journey if clues exist.
- If data is truly minimal, produce the best possible output without padding with fiction.

ROLE DETECTION: Identify their role first:
- "raising funding", "seeking investment", "CEO of a startup" → FOUNDER
- "investing", "portfolio", "check size", "angel investor" → INVESTOR
- "advisory", "consulting", "mentoring" → ADVISOR
- "recruiting", "talent acquisition", "staffing" → RECRUITER (not entrepreneur)
- "agency", "consulting firm", "service provider" → SERVICE PROVIDER
- "job search", "career change", "open to opportunities" → JOB SEEKER
- "partnership", "joint venture" → PARTNERSHIP SEEKER

Output STRICTLY valid JSON matching this schema:
{
  "persona": {
    "name": "string - Creative title (e.g. 'The Growth-Focused Founder')",
    "archetype": "string - Descriptive classification",
    "designation": "string - Job title from data or 'Not specified'",
    "experience": "string - Exact years/description or 'Not specified'",
    "focus": "string - Key areas separated by ' | ' — be specific",
    "profile_essence": "string - 8-12 sentences. Rich narrative: who they are, journey, achievements, expertise, working style, what drives them, what makes them unique. This is the MOST IMPORTANT field for matching.",
    "strategy": "string - 6-8 detailed bullet points with specifics",
    "what_theyre_looking_for": "string - 4-6 sentences about what they seek",
    "engagement_style": "string - 2-3 sentences about interaction preferences"
  },
  "requirements": "string - 6-8 sentences about what they ACTIVELY SEEK. Be specific about type of person, industry, stage, geography. Include WHY they need this.",
  "offerings": "string - 6-8 sentences about what they can PROVIDE. Be EXTREMELY specific: companies, metrics, networks, skills, achievements."
}

IMPORTANT: profile_essence + requirements + offerings are used for embedding vectors.
LONGER AND MORE DETAILED = BETTER MATCHES. Expand and elaborate, don't compress.

Your response MUST be ONLY valid JSON. Start with {{ and end with }}.

Combined Input Data:
{combined_data}"""


def get_db():
    return psycopg2.connect(DATABASE_URL)


def get_all_users(conn, single_user_id=None):
    """Fetch users with completed personas."""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    if single_user_id:
        cursor.execute("""
            SELECT user_id, raw_questions, resume_text, conversation_text,
                   persona_name, persona_requirements, persona_offerings,
                   persona_profile_essence, persona_archetype, persona_designation,
                   persona_experience, persona_focus, persona_strategy,
                   persona_what_looking_for, persona_engagement_style
            FROM user_profiles
            WHERE persona_status = 'completed' AND user_id = %s
        """, (single_user_id,))
    else:
        cursor.execute("""
            SELECT user_id, raw_questions, resume_text, conversation_text,
                   persona_name, persona_requirements, persona_offerings,
                   persona_profile_essence, persona_archetype, persona_designation,
                   persona_experience, persona_focus, persona_strategy,
                   persona_what_looking_for, persona_engagement_style
            FROM user_profiles
            WHERE persona_status = 'completed'
            ORDER BY created_at
        """)
    users = cursor.fetchall()
    cursor.close()
    return users


def build_combined_data(user: dict) -> str:
    """Build the combined data string from user's existing data."""
    parts = []

    # Full conversation text (if available — new users only)
    conv = user.get('conversation_text') or ''
    if conv.strip():
        parts.append("=" * 60)
        parts.append("FULL ONBOARDING CONVERSATION")
        parts.append("=" * 60)
        parts.append(conv.strip())
        parts.append("")

    # Q&A pairs
    raw_q = user.get('raw_questions') or []
    if isinstance(raw_q, str):
        try:
            raw_q = json.loads(raw_q)
        except:
            raw_q = []

    if raw_q:
        parts.append("=" * 60)
        parts.append("USER Q&A RESPONSES")
        parts.append("=" * 60)
        for i, q in enumerate(raw_q, 1):
            if isinstance(q, dict):
                prompt = q.get('prompt', '')
                answer = q.get('answer', '')
                parts.append(f"{i}. {prompt}\n   Answer: {answer}")
        parts.append("")

    # Resume
    resume = user.get('resume_text') or ''
    if resume.strip():
        parts.append("=" * 60)
        parts.append("RESUME / CV / PROFESSIONAL BACKGROUND")
        parts.append("=" * 60)
        parts.append(resume.strip())

    # If no conversation and no resume, include current persona as seed
    if not conv.strip() and not resume.strip():
        parts.append("=" * 60)
        parts.append("EXISTING PERSONA (expand on this)")
        parts.append("=" * 60)
        for field in ['persona_profile_essence', 'persona_requirements', 'persona_offerings',
                      'persona_strategy', 'persona_what_looking_for']:
            val = user.get(field) or ''
            if val and val != 'Not specified':
                label = field.replace('persona_', '').replace('_', ' ').title()
                parts.append(f"{label}: {val}")

    return "\n".join(parts)


def call_llm(combined_data: str) -> dict:
    """Call Sonnet 4.6 to generate enriched persona."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    prompt_text = ENRICHMENT_PROMPT.replace("{combined_data}", combined_data)

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt_text}]
    )

    content = response.content[0].text.strip()

    # Parse JSON
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    return json.loads(content.strip())


def update_user_persona(conn, user_id: str, persona_data: dict):
    """Write enriched persona back to database."""
    persona = persona_data.get('persona', {})
    requirements = persona_data.get('requirements', '')
    offerings = persona_data.get('offerings', '')

    # Ensure all values are strings
    def s(v):
        if v is None: return ''
        if isinstance(v, list): return '; '.join(str(i) for i in v)
        if isinstance(v, dict): return '; '.join(f"{k}: {v2}" for k, v2 in v.items())
        return str(v)

    cursor = conn.cursor()
    cursor.execute("""
        UPDATE user_profiles SET
            persona_name = %s,
            persona_archetype = %s,
            persona_designation = %s,
            persona_experience = %s,
            persona_focus = %s,
            persona_profile_essence = %s,
            persona_strategy = %s,
            persona_what_looking_for = %s,
            persona_engagement_style = %s,
            persona_requirements = %s,
            persona_offerings = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = %s
    """, (
        s(persona.get('name', '')),
        s(persona.get('archetype', '')),
        s(persona.get('designation', '')),
        s(persona.get('experience', '')),
        s(persona.get('focus', '')),
        s(persona.get('profile_essence', '')),
        s(persona.get('strategy', '')),
        s(persona.get('what_theyre_looking_for', '')),
        s(persona.get('engagement_style', '')),
        s(requirements),
        s(offerings),
        user_id,
    ))
    conn.commit()
    cursor.close()


def update_user_summary(conn, user_id: str, persona_data: dict):
    """Update the markdown summary in user_summaries table."""
    persona = persona_data.get('persona', {})
    requirements = persona_data.get('requirements', '')
    offerings = persona_data.get('offerings', '')

    def s(v):
        if v is None: return ''
        if isinstance(v, list): return '; '.join(str(i) for i in v)
        return str(v)

    parts = []
    if persona.get('name'):
        parts.append(f"# {persona['name']}")
    if persona.get('archetype'):
        parts.append(f"**Profile Type:** {persona['archetype']}")
    if persona.get('designation'):
        parts.append(f"**Designation:** {persona['designation']}")
    if persona.get('experience'):
        parts.append(f"**Experience:** {persona['experience']}")
    if persona.get('focus'):
        parts.append(f"\n## Focus\n{persona['focus']}")
    if persona.get('profile_essence'):
        parts.append(f"\n## Profile Essence\n{persona['profile_essence']}")
    if persona.get('strategy'):
        parts.append(f"\n## Strategy\n{s(persona['strategy'])}")
    if persona.get('what_theyre_looking_for'):
        parts.append(f"\n## Looking For\n{persona['what_theyre_looking_for']}")
    if persona.get('engagement_style'):
        parts.append(f"\n## Engagement Style\n{persona['engagement_style']}")
    if offerings:
        parts.append(f"\n## Offerings\n{s(offerings)}")

    markdown = "\n\n".join(parts)

    cursor = conn.cursor()
    cursor.execute("""
        UPDATE user_summaries SET
            summary = %s,
            status = 'approved',
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = %s
    """, (markdown, user_id))

    if cursor.rowcount == 0:
        # Insert if not exists
        cursor.execute("""
            INSERT INTO user_summaries (user_id, summary, status, urgency, created_at, updated_at)
            VALUES (%s, %s, 'approved', 'ongoing', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (user_id, markdown))

    conn.commit()
    cursor.close()


def count_words(persona_data: dict) -> int:
    """Count total words across persona fields."""
    total = 0
    persona = persona_data.get('persona', {})
    for v in persona.values():
        if isinstance(v, str):
            total += len(v.split())
    total += len(str(persona_data.get('requirements', '')).split())
    total += len(str(persona_data.get('offerings', '')).split())
    return total


def run(dry_run=True, single_user=None, batch_size=5, user_ids_filter=None):
    print(f"\n{'='*60}")
    print(f"ENRICH AI SUMMARIES — Step 1 of Embedding Upgrade")
    print(f"{'='*60}")
    print(f"Model:  {MODEL}")
    print(f"Mode:   {'DRY RUN' if dry_run else '*** LIVE ***'}")
    if single_user:
        print(f"User:   {single_user}")
    print(f"{'='*60}\n")

    conn = get_db()
    users = get_all_users(conn, single_user)
    print(f"Users with completed persona: {len(users)}\n")

    stats = {"total": len(users), "enriched": 0, "skipped": 0, "failed": 0}
    batch_count = 0

    for i, user in enumerate(users):
        uid = str(user['user_id'])
        name = user.get('persona_name') or uid[:8]

        # Check existing persona word count
        existing_words = 0
        for field in ['persona_profile_essence', 'persona_requirements', 'persona_offerings', 'persona_strategy']:
            val = user.get(field) or ''
            existing_words += len(val.split())

        combined = build_combined_data(user)
        input_chars = len(combined)

        if input_chars < 50:
            stats["skipped"] += 1
            logger.info(f"[{i+1}/{len(users)}] SKIP {name} — not enough input data ({input_chars} chars)")
            continue

        # Skip already-enriched users (>500 words = already processed by Sonnet 4.6)
        if existing_words > 500:
            stats["skipped"] += 1
            logger.info(f"[{i+1}/{len(users)}] SKIP {name} — already enriched ({existing_words} words)")
            continue

        if dry_run:
            logger.info(f"[{i+1}/{len(users)}] WOULD enrich {name} "
                       f"(current: ~{existing_words} words, input: {input_chars} chars)")
            stats["enriched"] += 1
            continue

        # Call LLM
        try:
            logger.info(f"[{i+1}/{len(users)}] Enriching {name}...")
            result = call_llm(combined)
            new_words = count_words(result)

            # Verify we got richer output
            if new_words < existing_words:
                logger.warning(f"  New output ({new_words} words) shorter than existing ({existing_words} words). Keeping new anyway.")

            # Update database — persona first (critical), summary second (optional)
            update_user_persona(conn, uid, result)
            # Summary is optional — don't let it roll back the persona update
            try:
                update_user_summary(conn, uid, result)
            except Exception as summary_err:
                logger.warning(f"  Summary update failed (persona saved OK): {summary_err}")
                try: conn.rollback()
                except: pass

            stats["enriched"] += 1
            logger.info(f"  OK: {existing_words} → {new_words} words "
                       f"({'+' if new_words > existing_words else ''}{new_words - existing_words})")

            # Rate limiting
            batch_count += 1
            if batch_count >= batch_size:
                batch_count = 0
                logger.info(f"  --- Batch pause (3s) ---")
                time.sleep(3)
            else:
                time.sleep(0.5)  # Small delay between each call

        except json.JSONDecodeError as e:
            stats["failed"] += 1
            logger.error(f"  FAIL (JSON parse): {e}")
            try: conn.rollback()
            except: pass
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"  FAIL: {e}")
            try: conn.rollback()
            except: pass

    conn.close()

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total users:  {stats['total']}")
    print(f"Enriched:     {stats['enriched']}")
    print(f"Skipped:      {stats['skipped']}")
    print(f"Failed:       {stats['failed']}")
    print(f"{'='*60}")

    if dry_run:
        print("\nDRY RUN complete. Run without --dry-run to execute.")
    else:
        print("\nStep 1 complete. Now run Step 2:")
        print("  python scripts/reembed_all_users.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich AI summaries for all users (Step 1)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--user", type=str, help="Single user ID to process")
    parser.add_argument("--file", type=str, help="File with user IDs (one per line)")
    parser.add_argument("--batch-size", type=int, default=5, help="Users per batch (default: 5)")
    args = parser.parse_args()

    user_ids_filter = None
    if args.file:
        with open(args.file) as f:
            user_ids_filter = set(l.strip() for l in f if l.strip())
        print(f"Filtering to {len(user_ids_filter)} users from {args.file}")

    run(dry_run=args.dry_run, single_user=args.user, batch_size=args.batch_size, user_ids_filter=user_ids_filter)
