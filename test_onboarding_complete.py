"""
Test complete onboarding flow for E2E users.
Tests the AI conversation and persona generation.
"""
import requests
import time

AI_SERVICE = "http://localhost:8000/api/v1"
BACKEND = "http://localhost:3000/api/v1"
API_KEY = "dev-api-key"
AI_HEADERS = {"Content-Type": "application/json", "X-API-KEY": API_KEY}

# Test messages simulating user responses
ONBOARDING_MESSAGES = [
    "I'm a tech founder building an AI startup. Looking for investors and advisors.",
    "We're pre-seed, targeting Series A in 6 months. Need $2M.",
    "Our focus is B2B SaaS for healthcare. I have 10 years in the industry.",
    "I'm based in London and prefer meeting investors who understand European markets.",
    "My goal is to find strategic investors who can also help with introductions to hospital networks."
]

def complete_onboarding_for_user(user_email: str, user_id: str):
    """Complete onboarding flow for a single user."""
    print(f"\n{'='*60}")
    print(f"ONBOARDING: {user_email}")
    print('='*60)

    results = {"user": user_email, "steps": {}}

    # Step 1: Start onboarding session
    print("\n[Step 1] Starting onboarding session...")
    try:
        r = requests.post(f"{AI_SERVICE}/onboarding/start",
                         headers=AI_HEADERS,
                         json={"user_id": user_id},
                         timeout=30)
        if r.status_code == 200:
            data = r.json()
            session_id = data.get("session_id")
            print(f"  [OK] Session: {session_id[:8]}...")
            print(f"  [AI] {data.get('greeting', '')[:100]}...")
            results["steps"]["1_start"] = "PASS"
        else:
            print(f"  [FAIL] {r.status_code}: {r.text[:100]}")
            results["steps"]["1_start"] = f"FAIL: {r.status_code}"
            return results
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["1_start"] = f"ERROR: {e}"
        return results

    # Step 2: Chat conversation (3-5 exchanges)
    print("\n[Step 2] Having conversation...")
    try:
        for i, msg in enumerate(ONBOARDING_MESSAGES[:3]):
            r = requests.post(f"{AI_SERVICE}/onboarding/chat",
                            headers=AI_HEADERS,
                            json={
                                "user_id": user_id,
                                "session_id": session_id,
                                "message": msg
                            },
                            timeout=60)
            if r.status_code == 200:
                data = r.json()
                progress = data.get("progress_percent", 0)
                response = data.get("response", data.get("ai_response", ""))[:80]
                print(f"  [{i+1}] Progress: {progress}% - AI: {response}...")
            else:
                print(f"  [{i+1}] FAIL: {r.status_code}")
            time.sleep(1)
        results["steps"]["2_chat"] = "PASS"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["2_chat"] = f"ERROR: {e}"

    # Step 3: Check progress
    print("\n[Step 3] Checking progress...")
    try:
        r = requests.get(f"{AI_SERVICE}/onboarding/progress/{session_id}",
                        headers=AI_HEADERS,
                        timeout=30)
        if r.status_code == 200:
            data = r.json()
            progress = data.get("progress_percent", data.get("progress", 0))
            print(f"  [OK] Progress: {progress}%")
            results["steps"]["3_progress"] = f"PASS ({progress}%)"
        else:
            print(f"  [FAIL] {r.status_code}: {r.text[:100]}")
            results["steps"]["3_progress"] = f"FAIL: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["3_progress"] = f"ERROR: {e}"

    # Step 4: Finalize onboarding
    print("\n[Step 4] Finalizing onboarding...")
    try:
        r = requests.post(f"{AI_SERVICE}/onboarding/finalize/{session_id}",
                         headers=AI_HEADERS,
                         json={
                             "user_id": user_id
                         },
                         timeout=60)
        if r.status_code in [200, 201]:
            data = r.json()
            summary = data.get("summary", data.get("persona", ""))
            if isinstance(summary, dict):
                summary = summary.get("summary", str(summary))
            print(f"  [OK] Summary: {str(summary)[:100]}...")
            results["steps"]["4_finalize"] = "PASS"
        else:
            print(f"  [FAIL] {r.status_code}: {r.text[:200]}")
            results["steps"]["4_finalize"] = f"FAIL: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["4_finalize"] = f"ERROR: {e}"

    # Step 5: Complete onboarding (creates DynamoDB profile)
    print("\n[Step 5] Completing onboarding (creating DynamoDB profile)...")
    try:
        r = requests.post(f"{AI_SERVICE}/onboarding/complete",
                         headers=AI_HEADERS,
                         json={
                             "session_id": session_id,
                             "user_id": user_id
                         },
                         timeout=60)
        if r.status_code in [200, 201]:
            data = r.json()
            profile_created = data.get("profile_created", False)
            task_id = data.get("persona_task_id", "")
            print(f"  [OK] Profile created: {profile_created}, task: {task_id[:8] if task_id else 'N/A'}...")
            results["steps"]["5_complete"] = f"PASS (profile: {profile_created})"
        else:
            print(f"  [FAIL] {r.status_code}: {r.text[:200]}")
            results["steps"]["5_complete"] = f"FAIL: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["5_complete"] = f"ERROR: {e}"

    # Wait for persona generation
    print("\n  Waiting for persona generation...")
    time.sleep(8)

    # Step 6: Approve summary (triggers embedding generation)
    print("\n[Step 6] Approving summary (triggers embeddings)...")
    try:
        r = requests.post(f"{AI_SERVICE}/user/approve-summary",
                         headers=AI_HEADERS,
                         json={"user_id": user_id},
                         timeout=60)
        if r.status_code in [200, 201, 202]:
            print(f"  [OK] Summary approved, embeddings queued")
            results["steps"]["6_approve"] = "PASS"
        else:
            print(f"  [INFO] {r.status_code}: {r.text[:100]}")
            results["steps"]["6_approve"] = f"INFO: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["6_approve"] = f"ERROR: {e}"

    # Wait for embeddings generation
    print("\n  Waiting for embeddings generation...")
    time.sleep(5)

    # Step 7: Get matches
    print("\n[Step 7] Getting matches...")
    try:
        r = requests.get(f"{AI_SERVICE}/matching/{user_id}/matches",
                        headers=AI_HEADERS,
                        timeout=30)
        if r.status_code == 200:
            data = r.json()
            matches = data.get("data", {}).get("matches", []) if isinstance(data.get("data"), dict) else data.get("matches", [])
            if not matches and isinstance(data, list):
                matches = data
            print(f"  [OK] Found {len(matches)} matches")
            results["steps"]["7_matches"] = f"PASS ({len(matches)} matches)"
        else:
            print(f"  [INFO] {r.status_code}: {r.text[:100]}")
            results["steps"]["7_matches"] = f"INFO: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["7_matches"] = f"ERROR: {e}"

    return results

def get_e2e_users():
    """Get E2E test users from database."""
    import subprocess
    result = subprocess.run([
        "docker", "exec", "reciprocity-postgres", "psql",
        "-U", "reciprocity_user", "-d", "reciprocity_ai", "-t", "-c",
        "SELECT id, email FROM users WHERE email LIKE 'test_%' AND is_email_verified = true ORDER BY created_at DESC LIMIT 5;"
    ], capture_output=True, text=True)

    users = []
    for line in result.stdout.strip().split('\n'):
        if '|' in line:
            parts = line.split('|')
            user_id = parts[0].strip()
            email = parts[1].strip()
            if user_id and email:
                users.append({"id": user_id, "email": email})
    return users

if __name__ == "__main__":
    print("="*60)
    print("RECIPROCITY - ONBOARDING COMPLETION TEST")
    print("="*60)

    users = get_e2e_users()
    print(f"\nFound {len(users)} E2E users")

    all_results = []
    for user in users[:3]:  # Test first 3 users
        result = complete_onboarding_for_user(user["email"], user["id"])
        all_results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_passed = 0
    total_steps = 0

    for result in all_results:
        steps = result["steps"]
        passed = len([s for s in steps.values() if "PASS" in str(s)])
        total = len(steps)
        total_passed += passed
        total_steps += total

        print(f"\n{result['user']}: {passed}/{total} steps passed")
        for step, status in steps.items():
            icon = "[OK]" if "PASS" in str(status) else "[X]"
            print(f"  {icon} {step}: {status}")

    print(f"\n{'='*60}")
    print(f"OVERALL: {total_passed}/{total_steps} steps passed ({100*total_passed//total_steps}%)")
