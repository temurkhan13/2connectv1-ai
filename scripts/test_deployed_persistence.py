"""
Standalone test for deployed BUG-021 and BUG-022 fixes.
Uses only HTTP requests - no module imports needed.
"""
import asyncio
import httpx
import os
from datetime import datetime

# Staging URLs
BACKEND_URL = "https://twoconnectv1-backend.onrender.com/api/v1"
AI_SERVICE_URL = "https://twoconnectv1-ai.onrender.com/api/v1"
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://hnvwzrynnerytsosefts.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Test user
TEST_EMAIL = f"test-{int(datetime.now().timestamp())}@2connect-test.com"
TEST_PASSWORD = "TestPass123!"

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


async def create_user():
    """Create test user via backend API."""
    print(f"\n{BLUE}=== Step 1: Creating Test User ==={RESET}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BACKEND_URL}/auth/signup",
            json={
                "email": TEST_EMAIL,
                "password": TEST_PASSWORD,
                "first_name": "Test",
                "last_name": "User"
            }
        )

        if response.status_code in [200, 201]:
            data = response.json()
            user_id = data.get("result", {}).get("user", {}).get("id")
            print(f"{GREEN}[OK] User created: {user_id}{RESET}")
            print(f"   Email: {TEST_EMAIL}")
            return user_id
        else:
            print(f"{RED}[FAIL] Status: {response.status_code} - {response.text}{RESET}")
            return None


async def start_session(user_id):
    """Start onboarding session."""
    print(f"\n{BLUE}=== Step 2: Starting Onboarding Session ==={RESET}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{AI_SERVICE_URL}/onboarding/start",
            json={"user_id": user_id}
        )

        if response.status_code in [200, 201]:
            data = response.json()
            session_id = data.get("session_id")
            print(f"{GREEN}[OK] Session started: {session_id}{RESET}")
            return session_id
        else:
            print(f"{RED}[FAIL] Failed: {response.status_code} - {response.text}{RESET}")
            return None


async def send_message(user_id, session_id, message):
    """Send message and return response."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{AI_SERVICE_URL}/onboarding/chat",
            json={
                "user_id": user_id,
                "message": message,
                "session_id": session_id
            }
        )

        if response.status_code in [200, 201]:
            data = response.json()

            print(f"\n{YELLOW}User:{RESET} {message}")
            print(f"{YELLOW}AI:{RESET} {data.get('ai_response', '')[:100]}...")
            print(f"{YELLOW}Progress:{RESET} {data.get('completion_percent', 0)}%")

            extracted = data.get('extracted_slots', {})
            if extracted:
                print(f"{YELLOW}Extracted:{RESET} {list(extracted.keys())}")

            return data
        else:
            print(f"{RED}[FAIL] Message failed: {response.status_code}{RESET}")
            print(f"   {response.text}")
            return None


async def check_supabase_slots(user_id):
    """Query Supabase directly to verify slots."""
    print(f"\n{BLUE}=== Checking Supabase ==={RESET}")

    if not SUPABASE_KEY:
        print(f"{RED}[FAIL] SUPABASE_SERVICE_ROLE_KEY not set{RESET}")
        return None

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/onboarding_answers",
            params={"user_id": f"eq.{user_id}"},
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
        )

        if response.status_code == 200:
            slots = response.json()
            print(f"{GREEN}[OK] Found {len(slots)} slots in Supabase{RESET}")

            for slot in slots:
                print(f"   - {slot['slot_name']}: {slot['value']}")

            return slots
        else:
            print(f"{RED}[FAIL] Query failed: {response.status_code}{RESET}")
            return None


async def run_test():
    """Run complete test."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}  TESTING BUG-021 + BUG-022 FIXES (DEPLOYED){RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")

    # Step 1: Create user
    user_id = await create_user()
    if not user_id:
        print(f"\n{RED}[FAIL] TEST FAILED: Could not create user{RESET}")
        return False

    # Step 2: Start session
    session_id = await start_session(user_id)
    if not session_id:
        print(f"\n{RED}[FAIL] TEST FAILED: Could not start session{RESET}")
        return False

    # Step 3: Send messages that extract slots
    print(f"\n{BLUE}=== Step 3: Extracting Slots ==={RESET}")

    messages = [
        "I'm a founder looking for Series A funding.",
        "My company is in fintech and we need $2M.",
        "We're looking for strategic investors to help with business development.",
        "Our timeline is urgent - we want to close in 3 months."
    ]

    for msg in messages:
        result = await send_message(user_id, session_id, msg)
        if not result:
            print(f"\n{RED}[FAIL] TEST FAILED: Message failed{RESET}")
            return False
        await asyncio.sleep(2)  # Small delay

    # Step 4: Verify persistence
    slots = await check_supabase_slots(user_id)

    if not slots or len(slots) == 0:
        print(f"\n{RED}[FAIL] TEST FAILED: No slots in Supabase (BUG-022 NOT FIXED){RESET}")
        return False

    # Step 5: Test re-extraction (BUG-021)
    print(f"\n{BLUE}=== Step 4: Testing Re-extraction (BUG-021) ==={RESET}")
    print(f"{YELLOW}Sending message to update existing slot...{RESET}")

    result = await send_message(
        user_id,
        session_id,
        "Actually, we need between $1.5M and $3M"
    )

    if not result:
        print(f"\n{RED}[FAIL] TEST FAILED: Re-extraction failed (409 error?){RESET}")
        return False

    print(f"{GREEN}[OK] Re-extraction succeeded (no 409 error){RESET}")

    # Final verification
    print(f"\n{BLUE}=== Final Verification ==={RESET}")
    final_slots = await check_supabase_slots(user_id)

    print(f"\n{BLUE}{'=' * 70}{RESET}")

    if final_slots and len(final_slots) > 0:
        print(f"{GREEN}[OK][OK][OK] ALL TESTS PASSED [OK][OK][OK]{RESET}")
        print(f"\n{GREEN}BUG-021 FIXED:{RESET} No 409 errors on re-extraction")
        print(f"{GREEN}BUG-022 FIXED:{RESET} Slots persisted successfully")
        print(f"{GREEN}Total slots saved:{RESET} {len(final_slots)}")
        print(f"\n{GREEN}Production-ready for slot persistence!{RESET}")
        return True
    else:
        print(f"{RED}[FAIL] TEST FAILED{RESET}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_test())
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n{RED}[FAIL] TEST CRASHED: {e}{RESET}")
        import traceback
        traceback.print_exc()
        exit(1)
