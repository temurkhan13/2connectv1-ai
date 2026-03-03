"""
Test enhanced extraction hints with real user messages.

Verifies that:
1. Implicit offerings are extracted (connections, achievements, experience)
2. Implicit requirements are extracted (goals → needs)
3. Question count limit prevents over-questioning
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load .env from reciprocity-ai root
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from app.services.llm_slot_extractor import LLMSlotExtractor
from app.services.context_manager import ContextManager, TurnType


async def test_alex_chen_scenario():
    """
    Test Alex Chen scenario - should extract implicit offerings.

    Alex said:
    - "warm intros to UCSF and Stanford" → offering: healthcare network introductions
    - "23% error reduction in pilot" → offering: proven results/case studies
    - "pilot customers at two hospitals" → offering: domain expertise
    """
    print("\n" + "="*80)
    print("TEST 1: Alex Chen - Implicit Offerings Extraction")
    print("="*80)

    extractor = LLMSlotExtractor()

    # Alex's message with implicit offerings
    alex_message = """
    I'm building an AI diagnostic tool for hospitals. We've had pilot customers at two
    hospitals and achieved a 23% error reduction in diagnostic accuracy. I have warm
    intros to UCSF and Stanford medical centers. I'm trying to navigate FDA regulatory
    approval while raising our seed round.
    """

    print(f"\n📝 User message:\n{alex_message.strip()}\n")

    result = extractor.extract_slots(
        user_message=alex_message,
        conversation_history=[],
        already_filled_slots={}
    )

    print("✅ EXTRACTED SLOTS:")
    for slot_name, slot_data in result.extracted_slots.items():
        print(f"\n  {slot_name}:")
        print(f"    Value: {slot_data.value}")
        print(f"    Confidence: {slot_data.confidence:.2f}")
        print(f"    Reasoning: {slot_data.reasoning}")

    # Check if offerings were extracted
    if "offerings" in result.extracted_slots:
        offerings = result.extracted_slots["offerings"].value
        print(f"\n✅ OFFERINGS EXTRACTED: {offerings}")

        # Verify implicit extraction
        implicit_markers = ["intro", "ucsf", "stanford", "23%", "pilot", "hospital"]
        found_markers = [m for m in implicit_markers if m.lower() in offerings.lower()]

        if len(found_markers) >= 3:
            print(f"✅ SUCCESS: Extracted {len(found_markers)} implicit offering markers: {found_markers}")
        else:
            print(f"❌ FAIL: Only found {len(found_markers)} markers (need 3+): {found_markers}")
    else:
        print("❌ FAIL: No offerings extracted")

    # Check if requirements were extracted from "trying to navigate FDA"
    if "requirements" in result.extracted_slots:
        requirements = result.extracted_slots["requirements"].value
        print(f"\n✅ REQUIREMENTS EXTRACTED: {requirements}")

        if "fda" in requirements.lower() or "regulatory" in requirements.lower():
            print("✅ SUCCESS: Extracted implicit requirement (FDA guidance)")
        else:
            print("⚠️ PARTIAL: Requirements extracted but no FDA/regulatory mention")
    else:
        print("⚠️ No requirements extracted")

    print("\n" + "="*80)


async def test_question_limit():
    """
    Test question count limit - should stop after 5 questions if minimum slots filled.
    """
    print("\n" + "="*80)
    print("TEST 2: Question Count Limit (Max 5 Questions)")
    print("="*80)

    context_mgr = ContextManager()

    # Create session
    session = context_mgr.create_session(user_id="test_user")
    session_id = session.session_id

    print(f"\n📋 Max questions configured: {context_mgr.max_questions}")

    # Simulate 6 questions being asked
    for i in range(6):
        # Add assistant question
        context_mgr.add_turn(
            session_id=session_id,
            turn_type=TurnType.ASSISTANT,
            content=f"Question {i+1}: What are your goals?"
        )

        # Add user response with some slot data
        if i == 0:
            response = "I'm a founder building an AI startup"
        elif i == 1:
            response = "I need funding and advisors"
        elif i == 2:
            response = "I can offer AI expertise and product insights"
        elif i == 3:
            response = "Focused on healthcare and fintech"
        else:
            response = f"Answer {i+1}"

        context_mgr.add_turn(
            session_id=session_id,
            turn_type=TurnType.USER,
            content=response
        )

        # Check if complete after this turn
        is_complete = context_mgr.is_complete(session_id)

        questions_so_far = i + 1
        print(f"\n  After question {questions_so_far}: is_complete = {is_complete}")

        if is_complete:
            print(f"  ✅ AUTO-COMPLETED after {questions_so_far} questions")

            # Verify it's because of question limit
            session_updated = context_mgr.get_session(session_id)
            filled_slots = [
                name for name, slot in session_updated.slots.items()
                if slot.status.value in ["filled", "confirmed"]
            ]
            print(f"  📊 Slots filled: {len(filled_slots)} → {filled_slots}")

            if questions_so_far == 5:
                print("  ✅ SUCCESS: Stopped at exactly 5 questions (as configured)")
            elif questions_so_far > 5:
                print(f"  ❌ FAIL: Should have stopped at 5, but continued to {questions_so_far}")
            else:
                print(f"  ⚠️ STOPPED EARLY: Completed at {questions_so_far} (may be due to 80%+ slots filled)")

            break

    print("\n" + "="*80)


async def test_alice_anderson_scenario():
    """
    Test Alice Anderson scenario - should extract more from single response.

    Alice gave a rich answer that should fill multiple slots at once.
    """
    print("\n" + "="*80)
    print("TEST 3: Alice Anderson - Multi-Slot Extraction from Single Response")
    print("="*80)

    extractor = LLMSlotExtractor()

    alice_message = """
    I'm building an EdTech platform for K-12 math education. We have 50 pilot schools
    using our adaptive learning system. I'm a former teacher with 10 years of classroom
    experience, and I've raised a pre-seed round. Now I'm looking for education-focused
    investors who understand the long sales cycles in K-12 and can introduce me to
    district superintendents.
    """

    print(f"\n📝 User message:\n{alice_message.strip()}\n")

    result = extractor.extract_slots(
        user_message=alice_message,
        conversation_history=[],
        already_filled_slots={}
    )

    print("✅ EXTRACTED SLOTS:")
    slots_extracted = 0
    for slot_name, slot_data in result.extracted_slots.items():
        slots_extracted += 1
        print(f"\n  {slot_name}:")
        print(f"    Value: {slot_data.value}")
        print(f"    Confidence: {slot_data.confidence:.2f}")

    print(f"\n📊 TOTAL SLOTS EXTRACTED: {slots_extracted}")

    # Should extract at least 6 slots:
    # - user_type (founder)
    # - industry_focus (EdTech, Education)
    # - company_stage (MVP or Product-Market Fit)
    # - experience_years (10 years)
    # - requirements (investors, district connections)
    # - offerings (teaching expertise, pilot results)

    expected_slots = ["user_type", "industry_focus", "requirements", "offerings"]
    found_expected = [s for s in expected_slots if s in result.extracted_slots]

    print(f"\n  Critical slots found: {len(found_expected)}/{len(expected_slots)}")
    print(f"  Slots: {found_expected}")

    if len(found_expected) >= 3:
        print(f"  ✅ SUCCESS: Extracted {len(found_expected)} critical slots from single response")
    else:
        print(f"  ❌ FAIL: Only extracted {len(found_expected)} slots (need 3+)")

    print("\n" + "="*80)


async def main():
    """Run all tests."""
    print("\n🧪 ENHANCED EXTRACTION TESTS")
    print("Testing implicit extraction hints and question limits\n")

    try:
        await test_alex_chen_scenario()
        await test_alice_anderson_scenario()
        await test_question_limit()

        print("\n✅ ALL TESTS COMPLETE")
        print("\nExpected improvements:")
        print("  1. Alex's implicit offerings (intros, results) extracted")
        print("  2. Alice's single response fills 3+ slots")
        print("  3. Question count stops at 5 (configurable)")

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
