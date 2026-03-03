"""
Simple LLM explanation test without database dependencies.
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Test directly with LLM service
from app.services.llm_service import LLMService


async def test_llm_explanation():
    """Test LLM explanation generation directly."""

    print("=" * 80)
    print("Testing LLM-Powered Match Explanations (Direct)")
    print("=" * 80)
    print()

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        print("   Set it in .env or environment variables")
        return

    print(f"✅ API Key configured: {api_key[:20]}...")
    print()

    # Initialize LLM service
    try:
        llm_service = LLMService()
        print(f"✅ LLM Service initialized")
        print(f"   Model: {llm_service.model}")
        print(f"   Temperature: {llm_service.temperature}")
    except Exception as e:
        print(f"❌ Failed to initialize LLM service: {e}")
        return

    print()
    print("-" * 80)
    print("Test Case: Investor → Founder Match")
    print("-" * 80)
    print()

    # Sample investor persona
    investor = {
        "name": "Sarah Chen",
        "user_type": "Early-Stage FinTech Investor",
        "designation": "Partner at Seed Ventures",
        "experience": "10 years in FinTech investing, 4 payment infrastructure deals",
        "industry": "Payment Infrastructure | B2B SaaS | LATAM Focus",
        "requirements": "Seeking innovative FinTech startups at Seed/Series A stage with early traction. Focus on payment infrastructure, B2B models, and LATAM expansion.",
        "offerings": "Invested in 4 payment infrastructure companies ($400K-$800K each). Portfolio includes 2 LATAM FinTech startups. Can provide introductions to bank partnerships.",
        "what_theyre_looking_for": "Payment infrastructure startups, B2B SaaS models, LATAM markets",
        "engagement_style": "Direct, data-driven, warm introductions preferred"
    }

    # Sample founder persona
    founder = {
        "name": "Carlos Mendez",
        "user_type": "Early-Stage Payment Infrastructure Founder",
        "designation": "CEO & Co-Founder",
        "experience": "2 years building payment solutions for banks",
        "industry": "Payment Infrastructure | B2B SaaS | LATAM Market",
        "requirements": "Seeking $500K seed funding from FinTech-focused investors with LATAM experience. Looking for advisors who've scaled B2B SaaS in regulated industries. Need bank partnership introductions.",
        "offerings": "Payment infrastructure expertise, early traction with 5 banks ($50K MRR), technical co-founder background, willingness to share learnings with other founders.",
        "what_theyre_looking_for": "Seed investors ($500K), FinTech advisors, bank partnership introductions",
        "engagement_style": "Direct, data-driven, prefers warm intros"
    }

    # Scores
    scores = {
        "req_to_off": 0.80,
        "off_to_req": 0.76,
        "industry_match": 0.85,
        "stage_match": 0.75,
        "geography_match": 0.70,
        "overall_score": 0.78
    }

    print("👤 Investor: Sarah Chen")
    print("   Partner at Seed Ventures")
    print("   Focus: Payment Infrastructure | B2B SaaS | LATAM")
    print()
    print("👤 Founder: Carlos Mendez")
    print("   CEO & Co-Founder")
    print("   Focus: Payment Infrastructure | B2B SaaS | LATAM Market")
    print()
    print("📊 Match Score: 78%")
    print()

    print("⏳ Calling Claude Haiku for explanation...")
    print()

    try:
        result = await llm_service.generate_match_explanation(
            investor, founder, scores
        )

        print("=" * 80)
        print("📊 LLM MATCH EXPLANATION")
        print("=" * 80)
        print()

        print("📝 Summary:")
        print(f"   {result.get('summary', 'N/A')}")
        print()

        print("✨ Synergy Areas:")
        for i, synergy in enumerate(result.get('synergy_areas', []), 1):
            print(f"   {i}. {synergy}")
        print()

        print("⚠️  Friction Points:")
        for i, friction in enumerate(result.get('friction_points', []), 1):
            print(f"   {i}. {friction}")
        print()

        print("💬 Talking Points:")
        for i, topic in enumerate(result.get('talking_points', []), 1):
            print(f"   {i}. {topic}")
        print()

        print("=" * 80)
        print()

        # Check for specificity
        print("🔍 QUALITY CHECK:")
        print()

        summary = result.get('summary', '')
        synergies = ' '.join(result.get('synergy_areas', []))
        all_text = f"{summary} {synergies}".lower()

        # Look for specific details
        good_indicators = ['$', 'k mrr', 'bank', 'latam', 'brazil', 'seed', '500k', 'payment']
        bad_indicators = ['industry match', 'mutually beneficial', 'general', 'expertise']

        found_good = [ind for ind in good_indicators if ind in all_text]
        found_bad = [ind for ind in bad_indicators if ind in all_text]

        if found_good:
            print(f"   ✅ Specific details cited: {', '.join(found_good)}")
        else:
            print("   ⚠️  No specific details found")

        if found_bad:
            print(f"   ⚠️  Generic phrases found: {', '.join(found_bad)}")
        else:
            print("   ✅ No generic phrases detected")

        print()
        print("✅ Test complete!")

    except Exception as e:
        print(f"❌ Error generating explanation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_llm_explanation())
