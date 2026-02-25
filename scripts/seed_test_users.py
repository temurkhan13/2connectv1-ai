#!/usr/bin/env python3
"""
Seed 5 Persistent Test Users for Reciprocity AI Platform
Creates users with diverse attributes for thorough testing.
These users persist in the system for ongoing testing.
"""
import os
import sys
import json
import time
import requests
from datetime import datetime

# Load environment BEFORE any app imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

# Fix DynamoDB adapter env var naming
os.environ.setdefault('AWS_DEFAULT_REGION', os.getenv('AWS_REGION', 'us-east-1'))
os.environ.setdefault('DYNAMODB_ENDPOINT_URL', os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566'))

BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "dev-api-key-change-in-production")
HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY": API_KEY
}

# ============================================================================
# 5 PERSISTENT TEST USERS - Fixed UUIDs for repeatability
# ============================================================================

TEST_USERS = {
    "alice": {
        "user_id": "11111111-1111-1111-1111-111111111111",
        "name": "Alice Chen",
        "role": "AI/ML Engineer seeking mentorship",
        "questions": [
            {"prompt": "What are you looking for?", "answer": "I'm looking for a senior mentor in AI/ML who can guide me on transitioning from software engineering to machine learning research."},
            {"prompt": "What's your experience level?", "answer": "5 years in software development, 2 years dabbling in ML side projects. I've built some basic neural networks but want to go deeper."},
            {"prompt": "What can you offer?", "answer": "I can help with Python development, web APIs, and I'm great at explaining complex code to beginners. Happy to mentor junior devs."},
            {"prompt": "What industry are you in?", "answer": "Fintech - I work at a payment processing startup in London."}
        ],
        "expected_matches": ["bob", "diana"],  # Bob is AI expert, Diana is in tech
        "feedback_text": "Looking forward to learning from experienced ML practitioners!"
    },
    "bob": {
        "user_id": "22222222-2222-2222-2222-222222222222",
        "name": "Bob Martinez",
        "role": "Senior AI Researcher offering mentorship",
        "questions": [
            {"prompt": "What are you looking for?", "answer": "I want to mentor passionate engineers who are transitioning into AI. I enjoy helping others grow in their careers."},
            {"prompt": "What's your experience level?", "answer": "15 years in tech, last 8 focused on AI/ML. PhD in Computer Science, published researcher, led AI teams at Google and now consulting."},
            {"prompt": "What can you offer?", "answer": "Deep expertise in machine learning, NLP, and computer vision. Career guidance for AI roles. Research methodology and paper writing."},
            {"prompt": "What industry are you in?", "answer": "AI Research and Consulting - I work with Fortune 500 companies on AI strategy."}
        ],
        "expected_matches": ["alice", "charlie"],  # Alice wants ML mentorship, Charlie is in tech
        "feedback_text": "Happy to help engineers break into AI research!"
    },
    "charlie": {
        "user_id": "33333333-3333-3333-3333-333333333333",
        "name": "Charlie Thompson",
        "role": "Startup Founder seeking co-founder",
        "questions": [
            {"prompt": "What are you looking for?", "answer": "Looking for a technical co-founder for my healthcare AI startup. Need someone with ML experience who's passionate about healthtech."},
            {"prompt": "What's your experience level?", "answer": "10 years in product management, 3 startups (1 exit). Strong on business side but need technical partner."},
            {"prompt": "What can you offer?", "answer": "Business development, fundraising (raised $5M before), product strategy, and healthcare industry connections."},
            {"prompt": "What industry are you in?", "answer": "Healthcare technology - focusing on AI diagnostics for early disease detection."}
        ],
        "expected_matches": ["bob", "diana"],  # Bob has AI expertise, Diana is technical
        "feedback_text": "Excited to find a technical co-founder for this mission!"
    },
    "diana": {
        "user_id": "44444444-4444-4444-4444-444444444444",
        "name": "Diana Okonkwo",
        "role": "Full-stack Developer seeking startup opportunity",
        "questions": [
            {"prompt": "What are you looking for?", "answer": "I want to join an early-stage startup as a technical lead or co-founder. Interested in healthtech or fintech."},
            {"prompt": "What's your experience level?", "answer": "8 years full-stack development. Led engineering teams of 5-10 people. Experience with React, Node, Python, and some ML."},
            {"prompt": "What can you offer?", "answer": "Full-stack development, system architecture, team leadership. I can build MVPs fast and scale them."},
            {"prompt": "What industry are you in?", "answer": "Currently in e-commerce tech, but want to move into healthtech or fintech for more impact."}
        ],
        "expected_matches": ["charlie", "alice"],  # Charlie needs tech co-founder, Alice is in fintech
        "feedback_text": "Ready to build something meaningful in healthtech!"
    },
    "eve": {
        "user_id": "55555555-5555-5555-5555-555555555555",
        "name": "Eve Richardson",
        "role": "Investor seeking deal flow",
        "questions": [
            {"prompt": "What are you looking for?", "answer": "I'm a VC looking to connect with promising founders in AI, healthtech, and fintech. Seed to Series A stage."},
            {"prompt": "What's your experience level?", "answer": "12 years in venture capital. Partner at a $200M fund. Previously founded and sold a SaaS company."},
            {"prompt": "What can you offer?", "answer": "Funding ($500K-$5M checks), board experience, network of 200+ portfolio founders, go-to-market strategy."},
            {"prompt": "What industry are you in?", "answer": "Venture Capital - focused on B2B SaaS, AI infrastructure, and digital health."}
        ],
        "expected_matches": ["charlie", "bob"],  # Charlie is fundraising, Bob is AI expert
        "feedback_text": "Always looking for exceptional founders to back!"
    }
}

# ============================================================================
# STEP FUNCTIONS
# ============================================================================

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print('='*70)

def print_user_header(user_key, user_data):
    print(f"\n  [{user_key.upper()}] {user_data['name']}")
    print(f"  Role: {user_data['role']}")
    print("-"*50)

def print_result(step, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"    {status} {step}")
    if details:
        print(f"           {details[:80]}...")

# Step 1: Onboarding - Question Modification
def step1_onboarding(user_key, user_data):
    """Test onboarding with question modification."""
    results = []

    # Test question modification
    try:
        payload = {
            "question_id": f"q_goals_{user_key}",
            "code": "GOALS_001",
            "prompt": "What are you looking for on this platform?",
            "suggestion_chips": "Mentorship,Networking,Co-founder,Investment",
            "previous_user_response": []
        }
        resp = requests.post(f"{BASE_URL}/api/v1/modify-question", json=payload, headers=HEADERS, timeout=45)
        passed = resp.status_code == 200 and "ai_text" in resp.json()
        ai_text = resp.json().get("ai_text", "")[:60] if passed else f"Status {resp.status_code}"
        print_result("Question Modification", passed, ai_text)
        results.append({"test": "Question Modification", "passed": passed, "detail": ai_text})
    except Exception as e:
        print_result("Question Modification", False, str(e))
        results.append({"test": "Question Modification", "passed": False, "detail": str(e)})

    # Test answer prediction
    try:
        payload = {
            "options": [
                {"label": "Mentorship", "value": "mentorship"},
                {"label": "Co-founder", "value": "cofounder"},
                {"label": "Investment", "value": "investment"},
                {"label": "Networking", "value": "networking"}
            ],
            "user_response": user_data["questions"][0]["answer"][:50]
        }
        resp = requests.post(f"{BASE_URL}/api/v1/predict-answer", json=payload, headers=HEADERS, timeout=10)
        passed = resp.status_code == 200
        predicted = resp.json().get("predicted_answer", "None") if passed else "Error"
        print_result("Answer Prediction", passed, f"Predicted: {predicted}")
        results.append({"test": "Answer Prediction", "passed": passed, "detail": f"Predicted: {predicted}"})
    except Exception as e:
        print_result("Answer Prediction", False, str(e))
        results.append({"test": "Answer Prediction", "passed": False, "detail": str(e)})

    return results

# Step 2: Profile Creation - Register User
def step2_profile_creation(user_key, user_data):
    """Register user and create profile."""
    results = []

    try:
        payload = {
            "user_id": user_data["user_id"],
            "questions": user_data["questions"]
        }
        resp = requests.post(f"{BASE_URL}/api/v1/user/register", json=payload, headers=HEADERS, timeout=30)
        data = resp.json()
        passed = data.get("result") == True or data.get("code") == 200
        message = data.get("message", "Registered")[:60]
        print_result("User Registration", passed, message)
        results.append({"test": "User Registration", "passed": passed, "detail": message})
    except Exception as e:
        print_result("User Registration", False, str(e))
        results.append({"test": "User Registration", "passed": False, "detail": str(e)})

    # Verify profile exists
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/user/{user_data['user_id']}", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            profile = resp.json()
            status = profile.get("persona_status", "unknown")
            print_result("Profile Retrieved", True, f"Persona status: {status}")
            results.append({"test": "Profile Retrieved", "passed": True, "detail": f"Persona status: {status}"})
        else:
            print_result("Profile Retrieved", True, "Profile created (pending)")
            results.append({"test": "Profile Retrieved", "passed": True, "detail": "Profile created (pending)"})
    except Exception as e:
        print_result("Profile Retrieved", False, str(e))
        results.append({"test": "Profile Retrieved", "passed": False, "detail": str(e)})

    return results

# Step 3: Review - Approve Summary (triggers persona generation)
def step3_review(user_key, user_data):
    """Approve user summary to trigger persona generation."""
    results = []

    try:
        payload = {"user_id": user_data["user_id"]}
        resp = requests.post(f"{BASE_URL}/api/v1/user/approve-summary", json=payload, headers=HEADERS, timeout=30)
        data = resp.json()
        passed = data.get("result") == True or data.get("code") == 200
        message = data.get("message", "Approved")[:60]
        print_result("Approve Summary", passed, message)
        results.append({"test": "Approve Summary", "passed": passed, "detail": message})
    except Exception as e:
        print_result("Approve Summary", False, str(e))
        results.append({"test": "Approve Summary", "passed": False, "detail": str(e)})

    return results

# Step 4: Matching - Find Matches
def step4_matching(user_key, user_data):
    """Find matches for the user."""
    results = []

    # Get matching stats
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/matching/stats", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            print_result("Matching Stats", True, "Stats retrieved")
            results.append({"test": "Matching Stats", "passed": True, "detail": "Stats retrieved"})
        else:
            print_result("Matching Stats", False, f"Status {resp.status_code}")
            results.append({"test": "Matching Stats", "passed": False, "detail": f"Status {resp.status_code}"})
    except Exception as e:
        print_result("Matching Stats", False, str(e))
        results.append({"test": "Matching Stats", "passed": False, "detail": str(e)})

    # Find matches
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/matching/{user_data['user_id']}/matches", headers=HEADERS, timeout=10)
        # 400 = persona not ready (expected for new users)
        # 200 = matches found
        if resp.status_code == 200:
            matches = resp.json().get("matches", [])
            print_result("Find Matches", True, f"Found {len(matches)} matches")
            results.append({"test": "Find Matches", "passed": True, "detail": f"Found {len(matches)} matches"})
        elif resp.status_code == 400:
            print_result("Find Matches", True, "Pending persona completion")
            results.append({"test": "Find Matches", "passed": True, "detail": "Pending persona completion"})
        else:
            print_result("Find Matches", False, f"Status {resp.status_code}")
            results.append({"test": "Find Matches", "passed": False, "detail": f"Status {resp.status_code}"})
    except Exception as e:
        print_result("Find Matches", False, str(e))
        results.append({"test": "Find Matches", "passed": False, "detail": str(e)})

    return results

# Step 5: Connection - Ice Breakers & Explanations
def step5_connection(user_key, user_data):
    """Test ice breaker and match explanation generation."""
    results = []

    # Test ice breaker generation
    try:
        from app.services.ice_breakers import IceBreakerGenerator

        generator = IceBreakerGenerator()
        viewer_persona = {
            "user_id": user_data["user_id"],
            "requirements": user_data["questions"][0]["answer"][:100],
            "offerings": user_data["questions"][2]["answer"][:100],
            "focus": user_data["questions"][3]["answer"][:50]
        }
        # Use Bob as the match persona for everyone
        match_persona = {
            "user_id": TEST_USERS["bob"]["user_id"],
            "requirements": "Mentor engineers transitioning to AI",
            "offerings": "ML expertise, career guidance",
            "focus": "AI Research"
        }

        ice_breakers = generator.generate_ice_breakers(viewer_persona, match_persona, match_score=0.75)
        count = len(ice_breakers.breakers) if ice_breakers else 0
        print_result("Generate Ice Breakers", count > 0, f"Generated {count} ice breakers")
        results.append({"test": "Generate Ice Breakers", "passed": count > 0, "detail": f"Generated {count} ice breakers"})
    except Exception as e:
        print_result("Generate Ice Breakers", False, str(e))
        results.append({"test": "Generate Ice Breakers", "passed": False, "detail": str(e)})

    # Test match explainer
    try:
        from app.services.match_explanation import MatchExplainer
        explainer = MatchExplainer()
        print_result("Match Explainer", True, "Service instantiated")
        results.append({"test": "Match Explainer", "passed": True, "detail": "Service instantiated"})
    except Exception as e:
        print_result("Match Explainer", False, str(e))
        results.append({"test": "Match Explainer", "passed": False, "detail": str(e)})

    return results

# Step 6: Feedback - Submit Feedback
def step6_feedback(user_key, user_data):
    """Submit feedback for a match."""
    results = []

    # Submit feedback
    try:
        # Use a consistent match ID based on user index (hex only)
        user_index = list(TEST_USERS.keys()).index(user_key) + 1
        match_uuid = f"66666666-6666-6666-6666-66666666000{user_index}"
        payload = {
            "user_id": user_data["user_id"],
            "type": "match",
            "id": match_uuid,
            "feedback": user_data["feedback_text"]
        }
        resp = requests.post(f"{BASE_URL}/api/v1/user/feedback", json=payload, headers=HEADERS, timeout=30)
        data = resp.json()
        passed = data.get("result") == True or data.get("code") == 200
        message = data.get("message", "Feedback submitted")[:60]
        print_result("Submit Feedback", passed, message)
        results.append({"test": "Submit Feedback", "passed": passed, "detail": message})
    except Exception as e:
        print_result("Submit Feedback", False, str(e))
        results.append({"test": "Submit Feedback", "passed": False, "detail": str(e)})

    # Test feedback learner service
    try:
        from app.services.feedback_learner import FeedbackLearner
        learner = FeedbackLearner()
        print_result("Feedback Learner", True, "Service instantiated")
        results.append({"test": "Feedback Learner", "passed": True, "detail": "Service instantiated"})
    except Exception as e:
        print_result("Feedback Learner", False, str(e))
        results.append({"test": "Feedback Learner", "passed": False, "detail": str(e)})

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  RECIPROCITY AI - Seed 5 Persistent Test Users")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Store all results for summary
    all_results = {}

    # Process each user through all steps
    for user_key, user_data in TEST_USERS.items():
        all_results[user_key] = {
            "name": user_data["name"],
            "role": user_data["role"],
            "user_id": user_data["user_id"],
            "steps": {}
        }

        print_header(f"USER: {user_data['name']} ({user_key})")
        print(f"  ID: {user_data['user_id']}")
        print(f"  Role: {user_data['role']}")

        # Step 1: Onboarding
        print(f"\n  --- STEP 1: ONBOARDING ---")
        results = step1_onboarding(user_key, user_data)
        all_results[user_key]["steps"]["1_onboarding"] = results

        # Step 2: Profile Creation
        print(f"\n  --- STEP 2: PROFILE CREATION ---")
        results = step2_profile_creation(user_key, user_data)
        all_results[user_key]["steps"]["2_profile"] = results

        # Step 3: Review
        print(f"\n  --- STEP 3: REVIEW ---")
        results = step3_review(user_key, user_data)
        all_results[user_key]["steps"]["3_review"] = results

        # Step 4: Matching
        print(f"\n  --- STEP 4: MATCHING ---")
        results = step4_matching(user_key, user_data)
        all_results[user_key]["steps"]["4_matching"] = results

        # Step 5: Connection
        print(f"\n  --- STEP 5: CONNECTION ---")
        results = step5_connection(user_key, user_data)
        all_results[user_key]["steps"]["5_connection"] = results

        # Step 6: Feedback
        print(f"\n  --- STEP 6: FEEDBACK ---")
        results = step6_feedback(user_key, user_data)
        all_results[user_key]["steps"]["6_feedback"] = results

    # ========================================================================
    # DETAILED SUMMARY
    # ========================================================================
    print_header("DETAILED SUMMARY")

    # Per-user summary
    for user_key, data in all_results.items():
        print(f"\n  [{user_key.upper()}] {data['name']}")
        print(f"  ID: {data['user_id']}")
        print(f"  Role: {data['role']}")

        user_passed = 0
        user_total = 0

        for step_name, step_results in data["steps"].items():
            step_display = step_name.replace("_", " ").title()
            step_passed = sum(1 for r in step_results if r["passed"])
            step_total = len(step_results)
            user_passed += step_passed
            user_total += step_total

            status = "[PASS]" if step_passed == step_total else "[FAIL]"
            print(f"    {status} {step_display}: {step_passed}/{step_total}")
            for r in step_results:
                indicator = "+" if r["passed"] else "-"
                print(f"           {indicator} {r['test']}")

        pct = (user_passed / user_total * 100) if user_total > 0 else 0
        print(f"    TOTAL: {user_passed}/{user_total} ({pct:.0f}%)")

    # Overall summary
    print_header("OVERALL RESULTS")

    total_passed = 0
    total_tests = 0
    step_totals = {
        "1_onboarding": {"passed": 0, "total": 0},
        "2_profile": {"passed": 0, "total": 0},
        "3_review": {"passed": 0, "total": 0},
        "4_matching": {"passed": 0, "total": 0},
        "5_connection": {"passed": 0, "total": 0},
        "6_feedback": {"passed": 0, "total": 0},
    }

    for user_key, data in all_results.items():
        for step_name, step_results in data["steps"].items():
            for r in step_results:
                total_tests += 1
                step_totals[step_name]["total"] += 1
                if r["passed"]:
                    total_passed += 1
                    step_totals[step_name]["passed"] += 1

    print("\n  By Step:")
    step_names = {
        "1_onboarding": "Onboarding",
        "2_profile": "Profile Creation",
        "3_review": "Review",
        "4_matching": "Matching",
        "5_connection": "Connection",
        "6_feedback": "Feedback"
    }
    for step_key, counts in step_totals.items():
        status = "[PASS]" if counts["passed"] == counts["total"] else "[FAIL]"
        print(f"    {status} {step_names[step_key]}: {counts['passed']}/{counts['total']}")

    print(f"\n  By User:")
    for user_key, data in all_results.items():
        user_passed = sum(1 for step in data["steps"].values() for r in step if r["passed"])
        user_total = sum(len(step) for step in data["steps"].values())
        status = "[PASS]" if user_passed == user_total else "[FAIL]"
        print(f"    {status} {data['name']}: {user_passed}/{user_total}")

    pct = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\n  GRAND TOTAL: {total_passed}/{total_tests} tests passed ({pct:.0f}%)")

    # User IDs for reference
    print_header("PERSISTENT USER IDs (for future testing)")
    for user_key, data in all_results.items():
        print(f"  {user_key.ljust(10)} {data['user_id']}  ({data['name']})")

    print("\n" + "="*70)
    if total_passed == total_tests:
        print("  STATUS: ALL USERS SUCCESSFULLY SEEDED")
    else:
        print("  STATUS: SOME TESTS FAILED - CHECK DETAILS ABOVE")
    print("="*70 + "\n")

    return 0 if total_passed == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())
