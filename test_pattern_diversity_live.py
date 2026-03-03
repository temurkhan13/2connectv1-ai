"""
Live API test for question pattern diversity.

Tests the actual onboarding API to verify pattern fixes are working.
"""

import requests
import json
from typing import List, Dict
import sys

# Import pattern detection from test suite
sys.path.append('tests')
from test_question_pattern_diversity import (
    extract_opener,
    detect_structure,
    detect_punctuation_pattern
)


# API Configuration
API_BASE_URL = "https://twoconnectv1-ai.onrender.com/api/v1"
# API_BASE_URL = "http://localhost:8000/api/v1"  # Use for local testing


def start_session(user_id: str) -> Dict:
    """Start a new onboarding session."""
    url = f"{API_BASE_URL}/onboarding/start"
    payload = {"user_id": user_id}

    print(f"\n🚀 Starting session for user: {user_id}")
    print(f"POST {url}")

    response = requests.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    print(f"✅ Session started: {data['session_id']}")
    print(f"📊 Progress: {data['progress_percent']}%")

    return data


def send_message(session_id: str, user_id: str, message: str) -> Dict:
    """Send a user message and get AI response."""
    url = f"{API_BASE_URL}/onboarding/chat"
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "message": message
    }

    print(f"\n💬 User: {message[:60]}...")
    print(f"POST {url}")

    response = requests.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    print(f"🤖 AI: {data['ai_response'][:80]}...")
    print(f"📊 Progress: {data['completion_percent']}%")
    print(f"📝 Slots extracted: {len(data['extracted_slots'])}")

    return data


def analyze_patterns(questions: List[str]) -> Dict:
    """Analyze pattern diversity in AI questions."""
    print("\n" + "="*80)
    print("📊 PATTERN ANALYSIS")
    print("="*80)

    # Extract patterns
    openers = [extract_opener(q) for q in questions]
    structures = [detect_structure(q) for q in questions]
    punctuation = [detect_punctuation_pattern(q) for q in questions]

    # Count patterns
    opener_counts = {}
    for opener in openers:
        opener_counts[opener] = opener_counts.get(opener, 0) + 1

    structure_counts = {}
    for structure in structures:
        structure_counts[structure] = structure_counts.get(structure, 0) + 1

    punctuation_counts = {}
    for punct in punctuation:
        punctuation_counts[punct] = punctuation_counts.get(punct, 0) + 1

    # Calculate metrics
    unique_openers = len(set(openers))
    unique_structures = len(set(structures))
    em_dash_count = sum(1 for q in questions if "—" in q or " — " in q)

    opener_diversity_pct = (unique_openers / len(openers) * 100) if openers else 0
    structure_diversity_pct = (unique_structures / len(structures) * 100) if structures else 0
    em_dash_pct = (em_dash_count / len(questions) * 100) if questions else 0

    # Display results
    print(f"\n📋 Questions analyzed: {len(questions)}")
    print(f"\n🔤 OPENERS:")
    for i, opener in enumerate(openers, 1):
        print(f"  Q{i}: {opener}")
    print(f"  Unique: {unique_openers}/{len(openers)} ({opener_diversity_pct:.1f}%)")
    print(f"  Distribution: {opener_counts}")

    print(f"\n🏗️ STRUCTURES:")
    for i, structure in enumerate(structures, 1):
        print(f"  Q{i}: {structure}")
    print(f"  Unique: {unique_structures}/{len(structures)} ({structure_diversity_pct:.1f}%)")
    print(f"  Distribution: {structure_counts}")

    print(f"\n✏️ PUNCTUATION:")
    for i, punct in enumerate(punctuation, 1):
        print(f"  Q{i}: {punct}")
    print(f"  Distribution: {punctuation_counts}")

    print(f"\n📏 EM DASH USAGE: {em_dash_count}/{len(questions)} questions ({em_dash_pct:.1f}%)")

    # Run tests
    print("\n" + "="*80)
    print("🧪 PATTERN DIVERSITY TESTS")
    print("="*80)

    tests_passed = 0
    tests_failed = 0

    # Test 1: No consecutive opener repetition
    consecutive_repetition = False
    for i in range(1, len(openers)):
        if openers[i] == openers[i-1]:
            print(f"\n❌ FAIL: Consecutive opener repetition")
            print(f"   Q{i} and Q{i+1} both use: {openers[i]}")
            consecutive_repetition = True
            tests_failed += 1
            break
    if not consecutive_repetition:
        print(f"\n✅ PASS: No consecutive opener repetition")
        tests_passed += 1

    # Test 2: Opener variety (no opener >40%)
    max_opener_pct = max(opener_counts.values()) / len(openers) * 100
    if max_opener_pct <= 40:
        print(f"✅ PASS: Opener variety sufficient ({max_opener_pct:.1f}% max)")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Opener variety too low ({max_opener_pct:.1f}% max, should be ≤40%)")
        tests_failed += 1

    # Test 3: Structural diversity (≥50% unique)
    if structure_diversity_pct >= 50:
        print(f"✅ PASS: Structural diversity sufficient ({structure_diversity_pct:.1f}%)")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Structural diversity too low ({structure_diversity_pct:.1f}%, should be ≥50%)")
        tests_failed += 1

    # Test 4: Em dash not overused (≤50%)
    if em_dash_pct <= 50:
        print(f"✅ PASS: Em dash usage acceptable ({em_dash_pct:.1f}%)")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Em dash overused ({em_dash_pct:.1f}%, should be ≤50%)")
        tests_failed += 1

    # Summary
    print("\n" + "="*80)
    print(f"📊 TEST SUMMARY: {tests_passed} PASSED, {tests_failed} FAILED")
    print("="*80)

    return {
        'openers': openers,
        'structures': structures,
        'punctuation': punctuation,
        'opener_diversity_pct': opener_diversity_pct,
        'structure_diversity_pct': structure_diversity_pct,
        'em_dash_pct': em_dash_pct,
        'tests_passed': tests_passed,
        'tests_failed': tests_failed
    }


def run_test_conversation():
    """Run a complete test conversation."""
    print("\n" + "="*80)
    print("🧪 LIVE ONBOARDING PATTERN DIVERSITY TEST")
    print("="*80)

    # Generate unique test user ID
    import uuid
    user_id = f"pattern_test_{uuid.uuid4().hex[:8]}"

    try:
        # Start session
        session_data = start_session(user_id)
        session_id = session_data['session_id']
        ai_questions = [session_data['greeting']]

        # Simulate user responses (founder profile)
        user_messages = [
            "Hi! I'm a software engineer with about 5 years of experience, currently working on an early-stage SaaS startup focused on productivity tools for remote teams. I'm looking to connect with other founders and entrepreneurs who've been through the early stages of building a product, as well as potential mentors who have experience in B2B SaaS. I'd also love to meet potential co-founders or technical collaborators who are passionate about the future of work.",

            "I'm pretty open to remote connections honestly — since I'm building tools for remote teams it kind of makes sense to embrace that myself. That said, I'm based in Austin, TX and I do love in-person meetups when possible. So ideally a mix of both — strong remote network globally, but also some local connections in the Austin startup scene.",

            "I can offer a lot on the technical side — things like system architecture, building scalable backends, choosing the right tech stack for early-stage products, and navigating engineering hiring. I've also picked up a fair bit about product development cycles and talking to customers to validate ideas. Happy to be a sounding board for anyone in the early stages of going from idea to MVP.",

            "Honestly, I get most excited about the pre-launch to early traction stage — that messy period where you're still figuring out if the thing you're building is actually what people want. That's where I think I can add the most value. Architecture decisions matter a lot at that stage and I've made enough mistakes myself to help others avoid common pitfalls. That said, I'd be curious to talk to people further along too, especially if they're dealing with team scaling challenges since that's something I'm starting to think about myself.",

            "I'm pre-revenue right now, actively working on MVP. Bootstrapping for now but will probably raise a small pre-seed round in the next 6 months. Team is just me at the moment, but looking to bring on a technical co-founder in the next few months."
        ]

        # Send messages and collect AI responses
        for i, message in enumerate(user_messages, 1):
            print(f"\n--- Turn {i} ---")
            response = send_message(session_id, user_id, message)

            ai_questions.append(response['ai_response'])

            # Stop if complete
            if response['is_complete']:
                print(f"\n✅ Onboarding complete at turn {i}")
                break

        # Display all questions
        print("\n" + "="*80)
        print("📝 ALL AI QUESTIONS")
        print("="*80)
        for i, question in enumerate(ai_questions, 0):
            if i == 0:
                print(f"\nQ0 (Greeting):\n{question}\n")
            else:
                print(f"\nQ{i}:\n{question}\n")

        # Analyze patterns (skip greeting)
        if len(ai_questions) > 1:
            analysis = analyze_patterns(ai_questions[1:])

            # Print verdict
            print("\n" + "="*80)
            if analysis['tests_failed'] == 0:
                print("🎉 SUCCESS! Pattern diversity fixes are WORKING!")
                print("✅ Questions feel unpredictable and human-like")
            else:
                print("⚠️ ISSUES DETECTED: Some patterns still present")
                print(f"❌ {analysis['tests_failed']} test(s) failed")
                print("🔧 May need additional tuning")
            print("="*80)

            return analysis
        else:
            print("\n⚠️ Not enough questions to analyze (need at least 2)")
            return None

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ API Error: {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n🚀 Starting live pattern diversity test...")
    print(f"🌐 API: {API_BASE_URL}")

    result = run_test_conversation()

    if result:
        # Exit code based on test results
        exit_code = 0 if result['tests_failed'] == 0 else 1
        print(f"\n🏁 Exit code: {exit_code}")
        exit(exit_code)
    else:
        print("\n🏁 Exit code: 1 (test failed to complete)")
        exit(1)
