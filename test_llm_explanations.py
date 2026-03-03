"""
Test script for LLM-powered match explanations.

Compares template-based vs LLM-generated explanations for real matches.
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from app.services.match_explanation import MatchExplainer
from app.services.multi_vector_matcher import MultiVectorMatch, MatchTier, DimensionScore
from app.adapters.dynamodb import UserProfile


async def test_llm_explanation():
    """Test LLM explanation generation with real user data."""

    print("=" * 80)
    print("Testing LLM-Powered Match Explanations")
    print("=" * 80)
    print()

    # Test user IDs - replace with real IDs from your system
    test_user_ids = [
        # Format: (viewer_id, match_id)
        # Add real user IDs here
    ]

    if not test_user_ids:
        print("⚠️  No test user IDs provided. Fetching from database...")
        # Try to get some real matches from the database
        try:
            from app.adapters.postgresql import postgresql_adapter
            matches = postgresql_adapter.get_recent_matches(limit=5)
            if matches:
                test_user_ids = [(m['user_id'], m['matched_user_id']) for m in matches[:2]]
                print(f"✅ Found {len(test_user_ids)} test pairs from database")
            else:
                print("❌ No matches found in database. Please provide user IDs manually.")
                return
        except Exception as e:
            print(f"❌ Could not fetch matches: {e}")
            return

    # Initialize explainer
    explainer = MatchExplainer()

    print(f"LLM Service Available: {explainer.use_llm}")
    print(f"Using: {'Claude Haiku (LLM)' if explainer.use_llm else 'Template-based fallback'}")
    print()

    for viewer_id, match_id in test_user_ids:
        print("-" * 80)
        print(f"Testing match: {viewer_id} → {match_id}")
        print("-" * 80)

        try:
            # Fetch personas
            viewer_profile = UserProfile.get(viewer_id)
            match_profile = UserProfile.get(match_id)

            viewer_persona = viewer_profile.persona.as_dict() if viewer_profile.persona else {}
            match_persona = match_profile.persona.as_dict() if match_profile.persona else {}

            # Add user_id to personas
            viewer_persona['user_id'] = viewer_id
            match_persona['user_id'] = match_id

            print(f"\n👤 Viewer: {viewer_persona.get('name', 'Unknown')}")
            print(f"   Role: {viewer_persona.get('archetype', 'Unknown')}")
            print(f"   Focus: {viewer_persona.get('focus', 'Not specified')[:80]}...")

            print(f"\n👤 Match: {match_persona.get('name', 'Unknown')}")
            print(f"   Role: {match_persona.get('archetype', 'Unknown')}")
            print(f"   Focus: {match_persona.get('focus', 'Not specified')[:80]}...")

            # Create mock match result (in production, this comes from matcher)
            mock_match = MultiVectorMatch(
                user_id=match_id,
                tier=MatchTier.STRONG,
                total_score=0.72,
                forward_score=0.75,
                reverse_score=0.69,
                dimension_scores={
                    "primary_goal": 0.85,
                    "industry_focus": 0.78,
                    "stage_preference": 0.65,
                    "geography": 0.60,
                    "engagement_style": 0.70,
                    "dealbreakers": 1.0
                }
            )

            # Generate explanation
            print("\n⏳ Generating explanation...")
            explanation = explainer.explain_match(
                match=mock_match,
                viewer_persona=viewer_persona,
                match_persona=match_persona
            )

            # Display results
            print("\n" + "=" * 80)
            print("📊 MATCH EXPLANATION")
            print("=" * 80)

            print(f"\n📝 Summary:")
            print(f"   {explanation.summary}")

            print(f"\n📖 Detailed Explanation:")
            print(f"   {explanation.detailed_explanation}")

            print(f"\n✨ Top Reasons:")
            for i, reason in enumerate(explanation.top_reasons, 1):
                print(f"   {i}. {reason}")

            if explanation.mutual_benefits:
                print(f"\n🤝 Mutual Benefits:")
                for i, benefit in enumerate(explanation.mutual_benefits, 1):
                    print(f"   {i}. {benefit.for_viewer}")

            if explanation.potential_concerns:
                print(f"\n⚠️  Potential Concerns:")
                for i, concern in enumerate(explanation.potential_concerns, 1):
                    print(f"   {i}. {concern}")

            print("\n" + "=" * 80)
            print()

        except Exception as e:
            print(f"❌ Error testing match {viewer_id} → {match_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n✅ Testing complete!")


def test_with_sample_data():
    """Test with hardcoded sample personas (for quick testing)."""

    print("=" * 80)
    print("Testing with Sample Data")
    print("=" * 80)

    # Sample investor persona
    investor_persona = {
        "user_id": "test_investor",
        "name": "Sarah Chen",
        "archetype": "Early-Stage FinTech Investor",
        "designation": "Partner at Seed Ventures",
        "experience": "10 years in FinTech investing, 4 payment infrastructure deals",
        "focus": "Payment Infrastructure | B2B SaaS | LATAM Focus",
        "profile_essence": "Series A investor focused on FinTech with expertise in payment infrastructure",
        "strategy": "Invest $400K-$800K in Seed/Series A FinTech companies",
        "what_theyre_looking_for": "Payment infrastructure startups, B2B SaaS models, LATAM markets",
        "engagement_style": "Direct, data-driven, warm introductions preferred",
        "requirements": "Seeking innovative FinTech startups at Seed/Series A stage with early traction. Focus on payment infrastructure, B2B models, and LATAM expansion.",
        "offerings": "Invested in 4 payment infrastructure companies ($400K-$800K each). Portfolio includes 2 LATAM FinTech startups. Can provide introductions to bank partnerships."
    }

    # Sample founder persona
    founder_persona = {
        "user_id": "test_founder",
        "name": "Carlos Mendez",
        "archetype": "Early-Stage Payment Infrastructure Founder",
        "designation": "CEO & Co-Founder",
        "experience": "2 years building payment solutions for banks",
        "focus": "Payment Infrastructure | B2B SaaS | LATAM Market",
        "profile_essence": "Building payment infrastructure for Brazilian banks",
        "strategy": "Focus on bank partnerships, B2B model, LATAM expansion starting with Brazil",
        "what_theyre_looking_for": "Seed investors ($500K), FinTech advisors, bank partnership introductions",
        "engagement_style": "Direct, data-driven, prefers warm intros",
        "requirements": "Seeking $500K seed funding from FinTech-focused investors with LATAM experience. Looking for advisors who've scaled B2B SaaS in regulated industries. Need bank partnership introductions.",
        "offerings": "Payment infrastructure expertise, early traction with 5 banks ($50K MRR), technical co-founder background, willingness to share learnings with other founders."
    }

    # Create mock match
    mock_match = MultiVectorMatch(
        user_id="test_founder",
        tier=MatchTier.STRONG,
        total_score=0.78,
        forward_score=0.80,
        reverse_score=0.76,
        dimension_scores={
            "primary_goal": 0.90,
            "industry_focus": 0.85,
            "stage_preference": 0.75,
            "geography": 0.70,
            "engagement_style": 0.80,
            "dealbreakers": 1.0
        }
    )

    # Generate explanation
    explainer = MatchExplainer()

    print(f"\nLLM Service Available: {explainer.use_llm}")
    print(f"Using: {'Claude Haiku' if explainer.use_llm else 'Templates'}\n")

    print("-" * 80)
    explanation = explainer.explain_match(
        match=mock_match,
        viewer_persona=investor_persona,
        match_persona=founder_persona
    )

    print("\n📊 MATCH EXPLANATION (Investor → Founder)")
    print("=" * 80)

    print(f"\n📝 Summary:\n   {explanation.summary}\n")
    print(f"📖 Detailed:\n   {explanation.detailed_explanation}\n")
    print("✨ Top Reasons:")
    for i, reason in enumerate(explanation.top_reasons, 1):
        print(f"   {i}. {reason}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM match explanations")
    parser.add_argument("--sample", action="store_true", help="Use sample data instead of real users")
    args = parser.parse_args()

    if args.sample:
        test_with_sample_data()
    else:
        asyncio.run(test_llm_explanation())
