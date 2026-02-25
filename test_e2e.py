# -*- coding: utf-8 -*-
import requests
import time
import json
import sys
import io
import uuid
from datetime import datetime

# Fix encoding on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://localhost:8000/api/v1"
HEADERS = {"X-API-KEY": "dev-api-key", "Content-Type": "application/json"}
USERS = [
    {"name": "Alice Chen", "role": "Founder", "industry": "FinTech", "objective": "fundraising", "message": "I am Alice, CEO of PayFlow, a fintech startup. We are raising our Series A to expand into Europe."},
    {"name": "Bob Smith", "role": "Investor", "industry": "Healthcare", "objective": "investing", "message": "Hi, I am Bob from Horizon Ventures. We invest in early-stage healthcare startups, typically Series A and B."},
    {"name": "Carol Davis", "role": "Startup Founder", "industry": "EdTech", "objective": "partnership", "message": "I am Carol, founder of LearnPath. We are looking for strategic partnerships with content providers."},
    {"name": "David Lee", "role": "Executive", "industry": "SaaS", "objective": "hiring", "message": "I am David, VP of Engineering at CloudScale. We are building out our AI team and need senior engineers."},
    {"name": "Eva Martinez", "role": "Entrepreneur", "industry": "Sustainability", "objective": "mentorship", "message": "I am Eva, working on a sustainability platform. Looking for mentors who have scaled green tech companies."},
]

def slots_to_questions(slots, user):
    """Convert extracted slots to question/answer format for registration."""
    questions = []

    # Map slot names to question prompts
    slot_to_prompt = {
        "primary_goal": "What is your primary goal?",
        "user_type": "What is your role?",
        "industry_focus": "What industries do you focus on?",
        "stage_preference": "What stage companies do you work with?",
        "check_size": "What is your typical check size?",
        "funding_need": "How much funding are you seeking?",
        "company_name": "What is your company name?",
        "expertise_areas": "What are your areas of expertise?",
    }

    for slot_name, slot_data in slots.items():
        if slot_name in slot_to_prompt:
            value = slot_data.get("value", slot_data) if isinstance(slot_data, dict) else slot_data
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            questions.append({
                "prompt": slot_to_prompt[slot_name],
                "answer": str(value)
            })

    # Add fallback questions if no slots extracted
    if not questions:
        questions = [
            {"prompt": "What is your primary goal?", "answer": user["objective"]},
            {"prompt": "What is your role?", "answer": user["role"]},
            {"prompt": "What industries do you focus on?", "answer": user["industry"]},
        ]

    return questions

def test_user_journey(user, user_num):
    print(f"\n{'='*60}")
    print(f"USER {user_num}: {user['name']} ({user['role']})")
    print(f"{'='*60}")
    results = {"user": user["name"], "steps": {}}
    session_id = None
    user_id = str(uuid.uuid4())
    all_slots = {}

    # Step 1: Start onboarding session
    print("\n[Step 1] Starting onboarding session...")
    try:
        r = requests.post(f"{BASE_URL}/onboarding/start", headers=HEADERS, json={
            "user_id": user_id,
            "objective": user["objective"]
        }, timeout=10)
        if r.status_code == 200:
            data = r.json()
            session_id = data.get("session_id")
            print(f"  [OK] Session created: {session_id[:16]}...")
            results["steps"]["1_start"] = "PASS"
        else:
            print(f"  [FAIL] {r.status_code} - {r.text[:100]}")
            results["steps"]["1_start"] = f"FAIL: {r.status_code}"
            return results
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["1_start"] = f"ERROR: {e}"
        return results

    # Step 2: Chat with AI (slot extraction)
    print("\n[Step 2] Sending chat message...")
    try:
        r = requests.post(f"{BASE_URL}/onboarding/chat", headers=HEADERS, json={
            "user_id": user_id,
            "session_id": session_id,
            "message": user["message"]
        }, timeout=30)
        if r.status_code == 200:
            data = r.json()
            slots = data.get("extracted_slots", {})
            all_slots = data.get("all_slots", {})
            print(f"  [OK] Extracted {len(slots)} new slots, {len(all_slots)} total")
            for slot, val in list(slots.items())[:3]:
                v = val.get('value', val) if isinstance(val, dict) else str(val)
                print(f"    - {slot}: {str(v)[:40]}")
            results["steps"]["2_chat"] = f"PASS ({len(slots)} slots)"
        else:
            print(f"  [FAIL] {r.status_code} - {r.text[:100]}")
            results["steps"]["2_chat"] = f"FAIL: {r.status_code}"
            return results
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["2_chat"] = f"ERROR: {e}"
        return results

    # Step 3: Check progress
    print("\n[Step 3] Checking progress...")
    try:
        r = requests.get(f"{BASE_URL}/onboarding/progress/{session_id}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            progress = data.get("progress_percent", 0)
            print(f"  [OK] Progress: {progress:.1f}%")
            results["steps"]["3_progress"] = f"PASS ({progress:.1f}%)"
        else:
            print(f"  [FAIL] {r.status_code}")
            results["steps"]["3_progress"] = f"FAIL: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["3_progress"] = f"ERROR: {e}"

    # Step 4: Finalize session
    print("\n[Step 4] Finalizing session...")
    try:
        r = requests.post(f"{BASE_URL}/onboarding/finalize/{session_id}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            collected = data.get("collected_data", {})
            slots_count = len(collected.get("slots", {}))
            print(f"  [OK] Session finalized with {slots_count} slots")
            results["steps"]["4_finalize"] = f"PASS ({slots_count} slots)"
        else:
            print(f"  [FAIL] {r.status_code} - {r.text[:100]}")
            results["steps"]["4_finalize"] = f"FAIL: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["4_finalize"] = f"ERROR: {e}"

    # Step 5: Register user with questions from slots
    print("\n[Step 5] Registering user...")
    try:
        questions = slots_to_questions(all_slots, user)
        r = requests.post(f"{BASE_URL}/user/register", headers=HEADERS, json={
            "user_id": user_id,
            "questions": questions
        }, timeout=60)
        if r.status_code in [200, 201, 202]:
            data = r.json()
            print(f"  [OK] User registered with {len(questions)} questions")
            results["steps"]["5_register"] = "PASS"
        else:
            print(f"  [FAIL] {r.status_code} - {r.text[:200]}")
            results["steps"]["5_register"] = f"FAIL: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["5_register"] = f"ERROR: {e}"

    # Give more time for async persona generation
    print("\n  Waiting for async persona generation...")
    time.sleep(5)

    # Step 6: Approve summary (triggers embedding generation)
    print("\n[Step 6] Approving summary...")
    try:
        r = requests.post(f"{BASE_URL}/user/approve-summary", headers=HEADERS, json={
            "user_id": user_id
        }, timeout=60)
        if r.status_code in [200, 201, 202]:
            print(f"  [OK] Summary approved")
            results["steps"]["6_approve"] = "PASS"
        else:
            print(f"  [FAIL] {r.status_code} - {r.text[:200]}")
            results["steps"]["6_approve"] = f"FAIL: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["6_approve"] = f"ERROR: {e}"

    # Wait for embedding generation
    time.sleep(3)

    # Step 7: Get user profile (check persona)
    print("\n[Step 7] Getting user profile...")
    try:
        r = requests.get(f"{BASE_URL}/user/{user_id}", headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            profile = data.get("data", data)
            has_persona = bool(profile.get("persona") or profile.get("ai_summary"))
            print(f"  [OK] Profile retrieved, has persona: {has_persona}")
            results["steps"]["7_profile"] = f"PASS (persona: {has_persona})"
        else:
            print(f"  [FAIL] {r.status_code} - {r.text[:100]}")
            results["steps"]["7_profile"] = f"FAIL: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["7_profile"] = f"ERROR: {e}"

    # Step 8: Get matches
    print("\n[Step 8] Getting matches...")
    try:
        r = requests.get(f"{BASE_URL}/matching/{user_id}/matches", headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            matches = data.get("data", {}).get("matches", []) if isinstance(data.get("data"), dict) else data.get("matches", [])
            if not matches and isinstance(data, list):
                matches = data
            print(f"  [OK] Found {len(matches)} matches")
            results["steps"]["8_matches"] = f"PASS ({len(matches)} matches)"
        else:
            print(f"  [FAIL] {r.status_code} - {r.text[:100]}")
            results["steps"]["8_matches"] = f"FAIL: {r.status_code}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["steps"]["8_matches"] = f"ERROR: {e}"

    return results

if __name__ == "__main__":
    # Run tests
    print("="*60)
    print("RECIPROCITY PLATFORM - FULL E2E TEST")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)

    all_results = []
    for i, user in enumerate(USERS, 1):
        result = test_user_journey(user, i)
        all_results.append(result)
        time.sleep(1)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_passed = 0
    total_steps = 0
    for result in all_results:
        user = result["user"]
        steps = result["steps"]
        passed = sum(1 for s in steps.values() if "PASS" in str(s))
        total = len(steps)
        total_passed += passed
        total_steps += total
        print(f"\n{user}: {passed}/{total} steps passed")
        for step, status in sorted(steps.items()):
            icon = "[OK]" if "PASS" in str(status) else "[X]"
            print(f"  {icon} {step}: {status}")

    print(f"\n{'='*60}")
    print(f"OVERALL: {total_passed}/{total_steps} steps passed ({100*total_passed/total_steps:.0f}%)")
    print(f"Completed: {datetime.now().isoformat()}")
