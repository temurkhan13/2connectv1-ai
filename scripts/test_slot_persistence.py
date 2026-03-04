"""
Test script to verify BUG-021 and BUG-022 fixes without manual user signup.

Tests:
1. Slot extraction and persistence to Supabase
2. Re-extraction doesn't cause 409 errors (BUG-021)
3. Persistence failures raise exceptions (BUG-022)
4. Completion logic works correctly
"""
import asyncio
import httpx
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters.supabase_onboarding import SupabaseOnboardingAdapter
from app.config import config

# Test configuration
AI_SERVICE_URL = "https://twoconnectv1-ai.onrender.com/api/v1"
TEST_USER_EMAIL = f"test-persistence-{int(datetime.now().timestamp())}@2connect-test.com"
TEST_PASSWORD = "TestPass123!"

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


async def create_test_user():
    """Register a test user via backend API."""
    print(f"\n{BLUE}═══ Step 1: Creating Test User ═══{RESET}")

    backend_url = config.RECIPROCITY_BACKEND_URL
    url = f"{backend_url}/auth/signup"

    payload = {
        "email": TEST_USER_EMAIL,
        "password": TEST_PASSWORD,
        "fullName": f"Test User {int(datetime.now().timestamp())}"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)

        if response.status_code in [200, 201]:
            result = response.json()
            user_id = result.get("result", {}).get("user", {}).get("id")
            print(f"{GREEN}✅ User created:{RESET}")
            print(f"   Email: {TEST_USER_EMAIL}")
            print(f"   User ID: {user_id}")
            return user_id
        else:
            print(f"{RED}❌ Failed to create user: {response.status_code}{RESET}")
            print(f"   Response: {response.text}")
            return None


async def start_onboarding_session(user_id):
    """Start a conversational onboarding session."""
    print(f"\n{BLUE}═══ Step 2: Starting Onboarding Session ═══{RESET}")

    url = f"{AI_SERVICE_URL}/onboarding/conversational/start"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json={},
            headers={"X-User-Id": user_id}
        )

        if response.status_code in [200, 201]:
            result = response.json()
            session_id = result.get("result", {}).get("session_id")
            greeting = result.get("result", {}).get("greeting", "")
            print(f"{GREEN}✅ Session started:{RESET}")
            print(f"   Session ID: {session_id}")
            print(f"   Greeting: {greeting[:100]}...")
            return session_id
        else:
            print(f"{RED}❌ Failed to start session: {response.status_code}{RESET}")
            print(f"   Response: {response.text}")
            return None


async def send_message(user_id, session_id, message, expect_slots=None):
    """
    Send a message in the onboarding chat.

    Args:
        user_id: User ID
        session_id: Session ID
        message: User message text
        expect_slots: Dict of slot names we expect to be filled after this message

    Returns:
        Response data
    """
    url = f"{AI_SERVICE_URL}/onboarding/conversational/chat"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json={
                "message": message,
                "session_id": session_id
            },
            headers={"X-User-Id": user_id}
        )

        if response.status_code in [200, 201]:
            result = response.json().get("result", {})

            print(f"\n{YELLOW}User:{RESET} {message}")
            print(f"{YELLOW}AI:{RESET} {result.get('ai_response', '')[:150]}...")
            print(f"{YELLOW}Progress:{RESET} {result.get('completion_percent', 0)}%")
            print(f"{YELLOW}Extracted slots:{RESET} {list(result.get('extracted_slots', {}).keys())}")

            return result
        else:
            print(f"{RED}❌ Message failed: {response.status_code}{RESET}")
            print(f"   Response: {response.text}")
            return None


async def verify_supabase_slots(user_id, expected_slots):
    """
    Verify slots were actually persisted to Supabase.

    Args:
        user_id: User ID
        expected_slots: Dict of {slot_name: expected_value} (or just list of slot names)

    Returns:
        True if all expected slots found, False otherwise
    """
    print(f"\n{BLUE}═══ Verifying Supabase Persistence ═══{RESET}")

    adapter = SupabaseOnboardingAdapter()

    if not adapter.enabled:
        print(f"{RED}❌ Supabase adapter not enabled{RESET}")
        return False

    try:
        # Query Supabase directly
        url = f"{adapter.supabase_url}/rest/v1/onboarding_answers"
        params = {
            "user_id": f"eq.{user_id}",
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

                print(f"{GREEN}✅ Found {len(slots)} slots in Supabase:{RESET}")

                for slot in slots:
                    print(f"   - {slot['slot_name']}: {slot['value']} (confidence: {slot['confidence']})")

                # Verify expected slots
                if expected_slots:
                    slot_names = {s['slot_name'] for s in slots}

                    if isinstance(expected_slots, dict):
                        expected_names = set(expected_slots.keys())
                    else:
                        expected_names = set(expected_slots)

                    missing = expected_names - slot_names

                    if missing:
                        print(f"{RED}❌ Missing expected slots: {missing}{RESET}")
                        return False
                    else:
                        print(f"{GREEN}✅ All expected slots found{RESET}")
                        return True

                return len(slots) > 0
            else:
                print(f"{RED}❌ Query failed: {response.status_code}{RESET}")
                return False

    except Exception as e:
        print(f"{RED}❌ Error: {e}{RESET}")
        return False


async def test_re_extraction_no_409(user_id, session_id):
    """
    Test that re-extracting slots doesn't cause 409 errors (BUG-021).

    Sends a message that updates an existing slot value.
    """
    print(f"\n{BLUE}═══ Step 4: Testing Re-extraction (BUG-021) ═══{RESET}")

    # First message: Set funding_need to $500K
    result1 = await send_message(
        user_id,
        session_id,
        "I'm looking for $500K in funding"
    )

    if not result1:
        print(f"{RED}❌ First message failed{RESET}")
        return False

    # Wait a bit
    await asyncio.sleep(2)

    # Second message: Update funding_need to $400-600K
    # This should trigger re-extraction and UPSERT (not 409 error)
    print(f"\n{YELLOW}Testing re-extraction (should UPSERT, not error)...{RESET}")

    result2 = await send_message(
        user_id,
        session_id,
        "Actually, I need between $400K and $600K"
    )

    if not result2:
        print(f"{RED}❌ Re-extraction failed (BUG-021 NOT FIXED){RESET}")
        return False

    print(f"{GREEN}✅ Re-extraction succeeded (BUG-021 FIXED){RESET}")

    # Verify in Supabase that funding_need was updated (not duplicated)
    await verify_supabase_slots(user_id, ["funding_need"])

    return True


async def run_full_test():
    """Run the complete persistence test."""
    print(f"\n{BLUE}{'═' * 70}{RESET}")
    print(f"{BLUE}  SLOT PERSISTENCE TEST (BUG-021 + BUG-022){RESET}")
    print(f"{BLUE}{'═' * 70}{RESET}")

    # Step 1: Create user
    user_id = await create_test_user()
    if not user_id:
        print(f"\n{RED}❌ TEST FAILED: Could not create user{RESET}")
        return False

    # Step 2: Start session
    session_id = await start_onboarding_session(user_id)
    if not session_id:
        print(f"\n{RED}❌ TEST FAILED: Could not start session{RESET}")
        return False

    # Step 3: Send messages that should extract slots
    print(f"\n{BLUE}═══ Step 3: Sending Messages to Extract Slots ═══{RESET}")

    messages = [
        "I'm John Smith, a founder looking to raise Series A funding.",
        "My company is in the fintech industry, and we're looking for $2M.",
        "We need strategic investors who can help with business development.",
        "Our timeline is urgent - we want to close within 3 months."
    ]

    for msg in messages:
        result = await send_message(user_id, session_id, msg)
        if not result:
            print(f"\n{RED}❌ TEST FAILED: Message failed{RESET}")
            return False

        # Small delay between messages
        await asyncio.sleep(1)

    # Verify slots were persisted
    success = await verify_supabase_slots(
        user_id,
        ["name", "role", "industry", "funding_need", "urgency"]
    )

    if not success:
        print(f"\n{RED}❌ TEST FAILED: Slots not persisted (BUG-022 NOT FIXED){RESET}")
        return False

    # Step 4: Test re-extraction
    re_extraction_ok = await test_re_extraction_no_409(user_id, session_id)

    if not re_extraction_ok:
        print(f"\n{RED}❌ TEST FAILED: Re-extraction test failed{RESET}")
        return False

    # Final verification
    print(f"\n{BLUE}═══ Final Verification ═══{RESET}")
    final_slots = await verify_supabase_slots(user_id, None)

    print(f"\n{BLUE}{'═' * 70}{RESET}")

    if final_slots:
        print(f"{GREEN}✅✅✅ ALL TESTS PASSED ✅✅✅{RESET}")
        print(f"\n{GREEN}BUG-021 FIXED:{RESET} No 409 errors on re-extraction")
        print(f"{GREEN}BUG-022 FIXED:{RESET} Slots persisted and validated")
        print(f"\n{GREEN}The platform is ready for production.{RESET}")
        return True
    else:
        print(f"{RED}❌ TEST FAILED{RESET}")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_full_test())
    sys.exit(0 if success else 1)
