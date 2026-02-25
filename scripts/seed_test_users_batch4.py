#!/usr/bin/env python3
"""
Reciprocity AI - Batch 4 Test Users (End-to-End Verification)

This batch tests the complete pipeline after all fixes:
1. User registration via backend API
2. Webhook sync to AI service
3. Persona generation via OpenAI
4. Embedding generation via SentenceTransformers
5. Matching via pgvector

5 Diverse Users:
- Alice: AI Startup Founder (seeks funding & technical talent)
- Bob: Angel Investor (seeks AI startups)
- Charlie: Full-Stack Developer (seeks mentorship & startup opportunities)
- Diana: Product Manager (seeks technical co-founders)
- Eve: Marketing Expert (seeks startup partnerships)

Date: February 2026
"""
import os
import sys
import requests
import time
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND_URL = "http://localhost:3000"
AI_SERVICE_URL = "http://localhost:8000"

# Generate unique timestamp for this batch
BATCH_TS = int(time.time())

# =============================================================================
# BATCH 4 TEST USERS - End-to-End Verification
# =============================================================================

TEST_USERS = [
    {
        "name": "Alice Chen",
        "email": f"alice.chen.{BATCH_TS}@test.com",
        "password": "Test123!@#",
        "gender": "female",
        "age": "32",
        "questions": [
            {"prompt": "Share your background", "answer": '{"bio": "AI startup founder with 8 years in machine learning. Previously led ML teams at Meta. Building next-gen computer vision platform.", "linkedin": "https://linkedin.com/in/alice-chen"}'},
            {"prompt": "Share your 20-second intro", "answer": "I'm building an AI-powered computer vision platform that helps retailers automate inventory management. Looking for seed funding and exceptional ML engineers."},
            {"prompt": "What's your #1 strength?", "answer": "Machine Learning & AI Architecture"},
            {"prompt": "What's another strength?", "answer": "Team building and technical leadership"},
            {"prompt": "One more strength?", "answer": "Fundraising and investor relations"},
            {"prompt": "What would your AI agent accomplish?", "answer": "Finding investors interested in AI/retail tech and top ML talent"},
            {"prompt": "Which stages do you prefer?", "answer": "Seed, Series A"},
            {"prompt": "Which industries matter most?", "answer": "AI, Retail Tech, Computer Vision"},
            {"prompt": "What roles to connect with?", "answer": "VCs, Angel Investors, Senior ML Engineers"},
            {"prompt": "Hard NO criteria?", "answer": "Non-technical investors, Crypto/Web3"},
            {"prompt": "How many introductions?", "answer": "Quality over Quantity"},
            {"prompt": "Communication style?", "answer": "Direct and professional"},
        ]
    },
    {
        "name": "Bob Martinez",
        "email": f"bob.martinez.{BATCH_TS}@test.com",
        "password": "Test123!@#",
        "gender": "male",
        "age": "48",
        "questions": [
            {"prompt": "Share your background", "answer": '{"bio": "Angel investor and former CTO. Made 25+ investments in AI/ML startups. 3 exits including 1 unicorn. Focused on early-stage AI companies.", "linkedin": "https://linkedin.com/in/bob-martinez"}'},
            {"prompt": "Share your 20-second intro", "answer": "I invest $50K-$200K in pre-seed and seed AI startups. Hands-on investor - I help with technical architecture, hiring, and enterprise sales introductions."},
            {"prompt": "What's your #1 strength?", "answer": "Technical due diligence and architecture review"},
            {"prompt": "What's another strength?", "answer": "Enterprise sales network and GTM strategy"},
            {"prompt": "One more strength?", "answer": "Recruiting senior engineering talent"},
            {"prompt": "What would your AI agent accomplish?", "answer": "Finding promising AI founders at pre-seed and seed stage"},
            {"prompt": "Which stages do you prefer?", "answer": "Pre-seed, Seed"},
            {"prompt": "Which industries matter most?", "answer": "AI, Enterprise SaaS, Developer Tools"},
            {"prompt": "What roles to connect with?", "answer": "Technical Founders, AI Researchers, CTOs"},
            {"prompt": "Hard NO criteria?", "answer": "Non-technical founders, B2C consumer apps"},
            {"prompt": "How many introductions?", "answer": "Quality over Quantity"},
            {"prompt": "Communication style?", "answer": "Direct and informal"},
        ]
    },
    {
        "name": "Charlie Wilson",
        "email": f"charlie.wilson.{BATCH_TS}@test.com",
        "password": "Test123!@#",
        "gender": "male",
        "age": "27",
        "questions": [
            {"prompt": "Share your background", "answer": '{"bio": "Full-stack developer with 5 years experience. Expert in React, Python, and cloud architecture. Looking to join an early-stage startup as technical co-founder or lead engineer.", "linkedin": "https://linkedin.com/in/charlie-wilson"}'},
            {"prompt": "Share your 20-second intro", "answer": "I'm a full-stack developer who loves building products from scratch. Seeking mentorship from experienced founders and opportunities to join a promising AI startup."},
            {"prompt": "What's your #1 strength?", "answer": "Full-stack development (React, Python, AWS)"},
            {"prompt": "What's another strength?", "answer": "Rapid prototyping and MVP development"},
            {"prompt": "One more strength?", "answer": "DevOps and cloud infrastructure"},
            {"prompt": "What would your AI agent accomplish?", "answer": "Finding mentors and early-stage startup opportunities"},
            {"prompt": "Which stages do you prefer?", "answer": "Pre-seed, Seed"},
            {"prompt": "Which industries matter most?", "answer": "AI, Developer Tools, SaaS"},
            {"prompt": "What roles to connect with?", "answer": "Startup Founders, CTOs, Senior Engineers"},
            {"prompt": "Hard NO criteria?", "answer": "Large corporations, non-technical roles"},
            {"prompt": "How many introductions?", "answer": "Moderate - a few good matches weekly"},
            {"prompt": "Communication style?", "answer": "Casual and conversational"},
        ]
    },
    {
        "name": "Diana Patel",
        "email": f"diana.patel.{BATCH_TS}@test.com",
        "password": "Test123!@#",
        "gender": "female",
        "age": "35",
        "questions": [
            {"prompt": "Share your background", "answer": '{"bio": "Product Manager with 10 years experience at Google and Stripe. Expert in fintech and payments. Looking for technical co-founder to build my startup idea.", "linkedin": "https://linkedin.com/in/diana-patel"}'},
            {"prompt": "Share your 20-second intro", "answer": "I have a validated fintech idea with early customer interest. Looking for a technical co-founder who can build the MVP while I handle product, sales, and fundraising."},
            {"prompt": "What's your #1 strength?", "answer": "Product strategy and roadmap planning"},
            {"prompt": "What's another strength?", "answer": "Enterprise sales and customer development"},
            {"prompt": "One more strength?", "answer": "Fundraising and investor pitching"},
            {"prompt": "What would your AI agent accomplish?", "answer": "Finding a technical co-founder for my fintech startup"},
            {"prompt": "Which stages do you prefer?", "answer": "Pre-seed, Seed"},
            {"prompt": "Which industries matter most?", "answer": "Fintech, Payments, Banking"},
            {"prompt": "What roles to connect with?", "answer": "Software Engineers, CTOs, Technical Co-founders"},
            {"prompt": "Hard NO criteria?", "answer": "People not interested in co-founding"},
            {"prompt": "How many introductions?", "answer": "Quality over Quantity"},
            {"prompt": "Communication style?", "answer": "Professional but friendly"},
        ]
    },
    {
        "name": "Eve Thompson",
        "email": f"eve.thompson.{BATCH_TS}@test.com",
        "password": "Test123!@#",
        "gender": "female",
        "age": "30",
        "questions": [
            {"prompt": "Share your background", "answer": '{"bio": "Growth marketing expert with 7 years in B2B SaaS. Helped 3 startups scale from $0 to $10M ARR. Looking to join an AI startup as head of marketing or CMO.", "linkedin": "https://linkedin.com/in/eve-thompson"}'},
            {"prompt": "Share your 20-second intro", "answer": "I'm a growth marketing expert who specializes in B2B SaaS. Looking for AI startups that need help with go-to-market strategy and customer acquisition."},
            {"prompt": "What's your #1 strength?", "answer": "B2B growth marketing and demand generation"},
            {"prompt": "What's another strength?", "answer": "Content marketing and SEO"},
            {"prompt": "One more strength?", "answer": "Marketing team building"},
            {"prompt": "What would your AI agent accomplish?", "answer": "Finding AI startups that need marketing leadership"},
            {"prompt": "Which stages do you prefer?", "answer": "Seed, Series A"},
            {"prompt": "Which industries matter most?", "answer": "AI, B2B SaaS, Enterprise Software"},
            {"prompt": "What roles to connect with?", "answer": "Startup Founders, CEOs, VPs of Sales"},
            {"prompt": "Hard NO criteria?", "answer": "B2C, consumer apps, crypto"},
            {"prompt": "How many introductions?", "answer": "Moderate - a few good matches weekly"},
            {"prompt": "Communication style?", "answer": "Casual and conversational"},
        ]
    }
]


def register_user_backend(user_data):
    """Register user via backend API."""
    # Step 1: Register
    register_payload = {
        "email": user_data["email"],
        "password": user_data["password"],
        "first_name": user_data["name"].split()[0],
        "last_name": user_data["name"].split()[-1]
    }

    resp = requests.post(f"{BACKEND_URL}/api/v1/auth/signup", json=register_payload)
    if resp.status_code not in [200, 201]:
        return None, f"Register failed: {resp.status_code} - {resp.text[:100]}"

    # Get verification code from response (available in non-production)
    signup_data = resp.json()
    verification_code = signup_data.get("result", {}).get("email_verification_code")

    # Step 2: Verify email
    if verification_code:
        verify_payload = {"email": user_data["email"], "code": verification_code}
        resp = requests.post(f"{BACKEND_URL}/api/v1/auth/verify-email", json=verify_payload)
        if resp.status_code not in [200, 201]:
            return None, f"Email verify failed: {resp.status_code}"

    # Step 3: Login
    login_payload = {"email": user_data["email"], "password": user_data["password"]}
    resp = requests.post(f"{BACKEND_URL}/api/v1/auth/signin", json=login_payload)
    if resp.status_code != 200:
        return None, f"Login failed: {resp.status_code}"

    data = resp.json()
    result = data.get("result", {})
    token = result.get("access_token") or result.get("accessToken")
    user_id = result.get("user", {}).get("id")

    if not token:
        return None, "No token in login response"

    return {"token": token, "user_id": user_id, "email": user_data["email"]}, None


def complete_onboarding(auth, user_data):
    """Complete onboarding by answering questions one by one."""
    headers = {"Authorization": f"Bearer {auth['token']}", "Content-Type": "application/json"}

    answer_index = 0
    max_questions = 40  # Safety limit (there are 32 questions)

    for _ in range(max_questions):
        # Get next question
        resp = requests.get(f"{BACKEND_URL}/api/v1/onboarding/question", headers=headers)
        if resp.status_code != 200:
            return f"Get question failed: {resp.status_code}"

        data = resp.json()
        # Question can be directly in result or nested
        result = data.get("result", {})
        question = result if result.get("id") else result.get("question")

        if not question or not question.get("id"):
            # No more questions - onboarding complete
            break

        question_id = question.get("id")
        input_type = question.get("input_type", "text")
        options = question.get("options", [])

        # Get answer - try to use test data or generate a reasonable default
        if answer_index < len(user_data["questions"]):
            answer = user_data["questions"][answer_index]["answer"]
        else:
            # Generate default answers based on input type
            if input_type == "single_select" and options:
                answer = options[0].get("value", "N/A")
            elif input_type == "multi_select" and options:
                answer = options[0].get("value", "N/A")
            else:
                answer = "N/A - Default answer"

        # Submit answer (ai_text is required by the DTO)
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/onboarding/submit-question",
            headers=headers,
            json={"question_id": question_id, "user_response": answer, "ai_text": ""}
        )

        if resp.status_code not in [200, 201]:
            return f"Submit question failed: {resp.status_code}"

        # Check if onboarding is complete
        submit_result = resp.json().get("result", {}).get("submitResponse", {})
        if submit_result.get("onboarding_status") == "completed":
            break

        answer_index += 1

    return None


def wait_for_persona(user_id, timeout=60):
    """Wait for persona generation to complete."""
    from app.adapters.dynamodb import UserProfile

    start = time.time()
    while time.time() - start < timeout:
        try:
            user = UserProfile.get(user_id)
            if user.persona_status == "completed":
                return user.persona.name if user.persona else "Unknown"
            elif user.persona_status == "failed":
                return None
        except UserProfile.DoesNotExist:
            pass
        time.sleep(2)

    return None


def main():
    print("=" * 70)
    print("BATCH 4: END-TO-END PIPELINE TEST (5 New Users)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Batch Timestamp: {BATCH_TS}")
    print()

    # Check services
    print("[0] Checking services...")
    try:
        requests.get(f"{BACKEND_URL}/health", timeout=5)
        print(f"    Backend ({BACKEND_URL}): OK")
    except:
        print(f"    Backend ({BACKEND_URL}): FAILED - Start backend first!")
        return

    try:
        requests.get(f"{AI_SERVICE_URL}/health", timeout=5)
        print(f"    AI Service ({AI_SERVICE_URL}): OK")
    except:
        print(f"    AI Service ({AI_SERVICE_URL}): FAILED - Start AI service first!")
        return

    print()
    print("[1] Registering users via backend...")
    print("-" * 70)

    registered = []
    for user_data in TEST_USERS:
        name = user_data["name"]
        print(f"    {name:20} ... ", end="", flush=True)

        auth, error = register_user_backend(user_data)
        if error:
            print(f"FAILED - {error}")
            continue

        print(f"Registered (ID: {auth['user_id'][:8]}...)")
        registered.append({"auth": auth, "data": user_data})

    print()
    print("[2] Completing onboarding...")
    print("-" * 70)

    onboarded = []
    for item in registered:
        name = item["data"]["name"]
        print(f"    {name:20} ... ", end="", flush=True)

        error = complete_onboarding(item["auth"], item["data"])
        if error:
            print(f"FAILED - {error}")
            continue

        print("Onboarded")
        onboarded.append(item)

    print()
    print("[3] Requesting AI summaries (triggers webhook to AI service)...")
    print("-" * 70)

    summaries_requested = []
    for item in onboarded:
        name = item["data"]["name"]
        headers = {"Authorization": f"Bearer {item['auth']['token']}", "Content-Type": "application/json"}
        print(f"    {name:20} ... ", end="", flush=True)

        # Request AI summary
        resp = requests.post(f"{BACKEND_URL}/api/v1/onboarding/request-ai-summary", headers=headers)
        if resp.status_code not in [200, 201]:
            print(f"FAILED - {resp.status_code}")
            continue

        print("Requested")
        summaries_requested.append(item)

    # Wait for AI summaries to be generated
    print()
    print("[4] Waiting for persona generation (via webhook -> OpenAI)...")
    print("-" * 70)

    # Load DynamoDB adapter
    from dotenv import load_dotenv
    load_dotenv(override=True)
    os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
    os.environ.setdefault('DYNAMODB_ENDPOINT_URL', 'http://localhost:4566')

    personas_ready = []
    for item in summaries_requested:
        name = item["data"]["name"]
        user_id = item["auth"]["user_id"]
        print(f"    {name:20} ... ", end="", flush=True)

        persona_name = wait_for_persona(user_id, timeout=90)
        if persona_name:
            print(f"OK - {persona_name[:30]}")
            personas_ready.append(item)
        else:
            print("TIMEOUT/FAILED")

    print()
    print("[5] Generating embeddings...")
    print("-" * 70)

    from app.services.embedding_service import embedding_service
    from app.adapters.dynamodb import UserProfile
    from app.adapters.postgresql import postgresql_adapter

    embeddings_ready = []
    for item in personas_ready:
        name = item["data"]["name"]
        user_id = item["auth"]["user_id"]
        print(f"    {name:20} ... ", end="", flush=True)

        try:
            user = UserProfile.get(user_id)
            persona = user.persona

            if persona and persona.requirements:
                req_text = " ".join(persona.requirements)
                req_vector = embedding_service.generate_embedding(req_text)
                if req_vector:
                    postgresql_adapter.store_embedding(user_id, 'requirements', req_vector, {})

            if persona and persona.offerings:
                off_text = " ".join(persona.offerings)
                off_vector = embedding_service.generate_embedding(off_text)
                if off_vector:
                    postgresql_adapter.store_embedding(user_id, 'offerings', off_vector, {})

            print("OK")
            embeddings_ready.append(item)
        except Exception as e:
            print(f"FAILED - {str(e)[:30]}")

    print()
    print("[6] Running matching algorithm...")
    print("-" * 70)

    from app.services.matching_service import matching_service

    for item in embeddings_ready:
        name = item["data"]["name"]
        user_id = item["auth"]["user_id"]
        print(f"    {name:20} ... ", end="", flush=True)

        try:
            result = matching_service.find_and_store_user_matches(user_id)
            total = result.get('total_matches', 0)
            print(f"OK - {total} matches")
        except Exception as e:
            print(f"FAILED - {str(e)[:30]}")

    print()
    print("=" * 70)
    print("BATCH 4 SUMMARY")
    print("=" * 70)
    print(f"Users registered:    {len(registered)}/5")
    print(f"Onboarding complete: {len(onboarded)}/5")
    print(f"Personas generated:  {len(personas_ready)}/5")
    print(f"Embeddings created:  {len(embeddings_ready)}/5")
    print(f"Matching complete:   {len(embeddings_ready)}/5")
    print("=" * 70)

    if embeddings_ready:
        print()
        print("[7] Sample matches for new users:")
        print("-" * 70)

        from app.adapters.dynamodb import UserMatches

        for item in embeddings_ready[:3]:
            name = item["data"]["name"]
            user_id = item["auth"]["user_id"]

            matches = UserMatches.get_user_matches(user_id)
            if matches:
                req_matches = matches.get('requirements_matches', [])[:2]
                print(f"\n    {name}:")
                for m in req_matches:
                    target = m.get('user_id', '?')[:20]
                    score = m.get('similarity_score', 0)
                    print(f"      -> {target}... (score: {score:.2f})")


if __name__ == "__main__":
    main()
