"""
Diagnostic script to check Henry Hall's data across all storage layers.

Shows exactly where data exists (or doesn't exist) and why.
"""
import asyncio
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters.supabase_onboarding import SupabaseOnboardingAdapter
from app.config import config
import httpx
import json

# Henry's info from dashboard screenshot
HENRY_USER_ID = "0ce69e26-7ad4-4bfa-9401-c079581e7354"
HENRY_EMAIL = "henry.hall.17725351i3@2connect-test.com"


async def check_supabase_onboarding_answers():
    """Check Supabase onboarding_answers table for Henry's slots."""
    print("\n" + "="*80)
    print("1. SUPABASE: onboarding_answers (Slot Storage)")
    print("="*80)

    adapter = SupabaseOnboardingAdapter()

    if not adapter.enabled:
        print("❌ Supabase adapter NOT ENABLED")
        print(f"   URL: {adapter.supabase_url}")
        print(f"   Key: {'SET' if adapter.supabase_key else 'NOT SET'}")
        return

    print(f"✅ Supabase adapter enabled")
    print(f"   URL: {adapter.supabase_url}")

    try:
        # Direct API query to see ALL data for Henry
        url = f"{adapter.supabase_url}/rest/v1/onboarding_answers"
        params = {
            "user_id": f"eq.{HENRY_USER_ID}",
            "select": "*"
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                url,
                params=params,
                headers=adapter._get_headers()
            )

            if response.status_code == 200:
                slots = response.json()
                print(f"\n📊 FOUND {len(slots)} SLOTS in Supabase:")

                if slots:
                    print("\n   Slot Details:")
                    for slot in slots:
                        print(f"   - {slot['slot_name']}: {slot['value']}")
                        print(f"     Confidence: {slot['confidence']}, Status: {slot['status']}")
                        print(f"     Created: {slot.get('created_at', 'N/A')}")
                else:
                    print("\n   ⚠️  ZERO SLOTS FOUND - Database is empty for this user!")
                    print("   This means all slot saves failed during onboarding.")
            else:
                print(f"❌ Query failed: {response.status_code}")
                print(f"   Response: {response.text}")

    except Exception as e:
        print(f"❌ Error querying Supabase: {e}")


async def check_supabase_user_embeddings():
    """Check Supabase user_embeddings table."""
    print("\n" + "="*80)
    print("2. SUPABASE: user_embeddings (Vector Storage)")
    print("="*80)

    supabase_url = config.RECIPROCITY_BACKEND_DB_URL
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials not configured")
        return

    try:
        url = f"{supabase_url}/rest/v1/user_embeddings"
        params = {"user_id": f"eq.{HENRY_USER_ID}"}
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)

            if response.status_code == 200:
                embeddings = response.json()
                print(f"\n📊 FOUND {len(embeddings)} EMBEDDINGS:")

                if embeddings:
                    for emb in embeddings:
                        print(f"   - Type: {emb.get('embedding_type')}")
                        print(f"     Vector dimensions: {len(emb.get('embedding', []))}")
                else:
                    print("\n   ⚠️  ZERO EMBEDDINGS - No vectors generated")
            else:
                print(f"❌ Query failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")


async def check_dynamodb_persona():
    """Check DynamoDB user_personas table."""
    print("\n" + "="*80)
    print("3. DYNAMODB: user_personas (Persona Storage)")
    print("="*80)

    try:
        import boto3
        from botocore.exceptions import ClientError

        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.Table('user_personas')

        response = table.get_item(Key={'user_id': HENRY_USER_ID})

        if 'Item' in response:
            persona = response['Item']
            print(f"\n📊 FOUND PERSONA:")
            print(f"   - Display name: {persona.get('display_name', 'N/A')}")
            print(f"   - Role: {persona.get('role', 'N/A')}")
            print(f"   - Created: {persona.get('created_at', 'N/A')}")
        else:
            print("\n   ⚠️  NO PERSONA FOUND in DynamoDB")
            print("   Persona generation failed (likely due to missing slots)")

    except ClientError as e:
        print(f"❌ DynamoDB error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def check_backend_database():
    """Check backend database (PostgreSQL) for user record."""
    print("\n" + "="*80)
    print("4. BACKEND DATABASE: users table")
    print("="*80)

    # This would require connecting to the backend's PostgreSQL
    # For now, we know from the dashboard that:
    print("\n📊 From Admin Dashboard:")
    print(f"   - User exists: ✅ {HENRY_EMAIL}")
    print(f"   - Onboarding status: completed")
    print(f"   - Slots: 0/11 ❌")
    print(f"   - Persona: no_profile ❌")


async def analyze_failure_chain():
    """Analyze why the data pipeline failed."""
    print("\n" + "="*80)
    print("5. FAILURE CHAIN ANALYSIS")
    print("="*80)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│ Expected Flow:                                                  │
├─────────────────────────────────────────────────────────────────┤
│ 1. User sends message                          ✅ WORKED       │
│ 2. AI extracts slots (in-memory)              ✅ WORKED       │
│ 3. Slots saved to Supabase                    ❌ FAILED       │
│    → 409 Conflict: funding_need already exists                 │
│    → Batch save returns 0                                      │
│    → NO EXCEPTION RAISED (silent failure)                      │
│ 4. Progress calculated from in-memory          ✅ 80%          │
│ 5. Completion check: 0 db slots < 80%         ❌ NOT COMPLETE  │
│ 6. No completion → No persona                 ❌ BLOCKED       │
│ 7. No persona → No embeddings                 ❌ BLOCKED       │
│ 8. No embeddings → No matches                 ❌ BLOCKED       │
└─────────────────────────────────────────────────────────────────┘

ROOT CAUSE:
- Step 3 failed with 409 error (duplicate key)
- BUG-021 fix (upsert) deployed AFTER Henry's session
- Error was logged but not raised → silent failure
- Frontend showed 80% (in-memory) but database had 0%
- User lost all data with no error message
    """)


async def main():
    """Run all diagnostic checks."""
    print("\n" + "="*80)
    print(f"DIAGNOSTIC: Henry Hall's Data Pipeline")
    print(f"User ID: {HENRY_USER_ID}")
    print(f"Email: {HENRY_EMAIL}")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*80)

    await check_supabase_onboarding_answers()
    await check_supabase_user_embeddings()
    await check_dynamodb_persona()
    await check_backend_database()
    await analyze_failure_chain()

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. ✅ BUG-021 already fixed (4a400b9) - Future sessions won't hit 409
2. 🔧 Need to implement retry + validation (prevent other failures)
3. 🔄 Need to recover Henry's data (3 options):

   Option A: Manual reconstruction from conversation transcript
   Option B: Ask Henry to redo onboarding (with fixes in place)
   Option C: Backfill from cached session data (if exists)
    """)


if __name__ == "__main__":
    asyncio.run(main())
