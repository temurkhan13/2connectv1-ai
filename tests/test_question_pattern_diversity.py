"""
Test suite for question pattern diversity in conversational onboarding.

Ensures AI doesn't follow detectable patterns that make it seem robotic or scripted.
Critical for maintaining user trust and perception of intelligence.
"""

import pytest
import re
from typing import List, Dict


def extract_opener(question: str) -> str:
    """Extract the opening phrase pattern from a question."""
    question = question.strip()
    words = question.split()[:5]

    # Normalize common patterns
    if question.startswith("That's"):
        return f"That's {words[1]}" if len(words) > 1 else "That's"
    elif question.startswith("What a"):
        return f"What a {words[2]}" if len(words) > 2 else "What a"
    elif question.startswith("What"):
        return "What [question]"
    elif question.startswith("How"):
        return "How [question]"
    else:
        return ' '.join(words[:3])


def detect_structure(question: str) -> str:
    """Detect the structural pattern of a question."""
    question = question.strip()

    # Em dash patterns
    if "—" in question or " — " in question:
        if any(bridge in question for bridge in ["As you", "Since you", "When you", "Given that"]):
            return "ack_em_dash_bridge_q"
        else:
            return "ack_em_dash_direct_q"

    # Bridge phrases without em dash
    if any(bridge in question for bridge in ["As you're", "Since you're", "When you think", "Given that you"]):
        return "ack_bridge_q"

    # Parenthetical
    if "(" in question and ")" in question:
        return "direct_q_parenthetical"

    # Two-part question
    if "?" in question[:-1]:
        return "two_part_q"

    return "direct_q"


def detect_punctuation_pattern(question: str) -> str:
    """Detect primary punctuation pattern."""
    if "—" in question or " — " in question:
        return "em_dash"
    elif ":" in question:
        return "colon"
    elif "(" in question:
        return "parenthetical"
    elif "," in question and question.count(",") >= 2:
        return "multi_comma"
    else:
        return "simple"


class TestQuestionPatternDiversity:
    """Test that AI questions maintain structural diversity."""

    @pytest.fixture
    def sample_conversation_questions(self) -> List[str]:
        """Sample questions from a typical onboarding conversation."""
        return [
            "Welcome to 2Connect! I'm here to help you find great connections. Tell me about yourself - what you're working on and what kind of connections you're looking for.",
            "That's impressive that you're building in the productivity space with 5 years of engineering under your belt — that's exactly the kind of domain expertise that makes for strong founding teams. As you're connecting with other founders and potential collaborators, are you mostly looking within a specific region or city, or are you open to building relationships remotely given the nature of what you're working on?",
            "That's a smart move — practicing what you preach by building remotely while creating remote team tools. Since you're looking to connect with both technical collaborators and potential mentors in the Austin scene and beyond, what kinds of expertise or support would be most valuable for you to offer back to your network right now?",
            "That's impressive — the combination of deep technical chops plus the customer validation piece is pretty rare. When you think about the founders you'd be most energized to work with, what does their situation typically look like?"
        ]

    @pytest.fixture
    def diverse_conversation_questions(self) -> List[str]:
        """Example of properly diverse questions."""
        return [
            "Welcome to 2Connect! Tell me about yourself - what you're working on and what kind of connections you're looking for.",
            "Building in the productivity space with 5 years of engineering expertise - that's powerful. Are you focusing on a specific region, or embracing remote connections?",
            "You mentioned being open to remote work. What would be most valuable for you to offer to collaborators right now? Technical guidance, customer insights, or something else?",
            "When you think about ideal founders to connect with, what stage are they typically at? Pre-launch wrestling with architecture, or further along hitting scaling challenges?"
        ]

    def test_no_consecutive_opener_repetition(self, sample_conversation_questions):
        """Test that consecutive questions don't use identical openers."""
        questions = sample_conversation_questions[1:]  # Skip welcome message

        openers = [extract_opener(q) for q in questions]

        # Check for consecutive repetitions
        for i in range(1, len(openers)):
            assert openers[i] != openers[i-1], (
                f"Consecutive questions have identical openers:\n"
                f"Q{i}: {openers[i-1]}\n"
                f"Q{i+1}: {openers[i]}\n"
                f"User will notice the pattern!"
            )

    def test_opener_variety_threshold(self, sample_conversation_questions):
        """Test that openers have sufficient variety (no opener used >40% of time)."""
        questions = sample_conversation_questions[1:]  # Skip welcome message

        if len(questions) < 3:
            pytest.skip("Need at least 3 questions to test variety")

        openers = [extract_opener(q) for q in questions]
        opener_counts = {}

        for opener in openers:
            opener_counts[opener] = opener_counts.get(opener, 0) + 1

        max_usage_pct = max(opener_counts.values()) / len(openers) * 100

        assert max_usage_pct <= 40, (
            f"Opener repetition too high: {max_usage_pct:.1f}%\n"
            f"Distribution: {opener_counts}\n"
            f"Users will perceive this as formulaic!"
        )

    def test_structural_diversity(self, sample_conversation_questions):
        """Test that sentence structures vary across questions."""
        questions = sample_conversation_questions[1:]  # Skip welcome message

        if len(questions) < 3:
            pytest.skip("Need at least 3 questions to test structure")

        structures = [detect_structure(q) for q in questions]

        # Check that not all questions use the same structure
        unique_structures = len(set(structures))
        structure_diversity_pct = unique_structures / len(structures) * 100

        assert structure_diversity_pct >= 50, (
            f"Structural diversity too low: {structure_diversity_pct:.1f}%\n"
            f"Structures used: {structures}\n"
            f"Questions feel templated!"
        )

    def test_em_dash_overuse(self, sample_conversation_questions):
        """Test that em dash isn't overused (max 50% of questions)."""
        questions = sample_conversation_questions[1:]  # Skip welcome message

        em_dash_count = sum(1 for q in questions if "—" in q or " — " in q)
        em_dash_pct = em_dash_count / len(questions) * 100

        assert em_dash_pct <= 50, (
            f"Em dash overused: {em_dash_pct:.1f}% of questions\n"
            f"Used in {em_dash_count}/{len(questions)} questions\n"
            f"Creates rhythmic monotony!"
        )

    def test_bridge_phrase_variety(self, sample_conversation_questions):
        """Test that bridge phrases (As you, Since you, When you) vary."""
        questions = sample_conversation_questions[1:]  # Skip welcome message

        bridge_phrases = ["As you", "Since you", "When you", "Given that"]
        bridge_counts = {phrase: sum(1 for q in questions if phrase in q) for phrase in bridge_phrases}

        # No single bridge phrase should be used in >40% of questions
        total_bridge_usage = sum(bridge_counts.values())

        if total_bridge_usage > 0:
            max_bridge_pct = max(bridge_counts.values()) / len(questions) * 100

            assert max_bridge_pct <= 40, (
                f"Bridge phrase overused: {max_bridge_pct:.1f}%\n"
                f"Distribution: {bridge_counts}\n"
                f"Pattern is too predictable!"
            )

    def test_diverse_questions_pass(self, diverse_conversation_questions):
        """Test that properly diverse questions pass all checks."""
        questions = diverse_conversation_questions[1:]  # Skip welcome message

        # Test opener variety
        openers = [extract_opener(q) for q in questions]
        for i in range(1, len(openers)):
            assert openers[i] != openers[i-1], "Consecutive openers should not match"

        # Test structural variety
        structures = [detect_structure(q) for q in questions]
        unique_structures = len(set(structures))
        assert unique_structures >= len(structures) * 0.5, "Should have 50%+ unique structures"

        # Test em dash usage
        em_dash_count = sum(1 for q in questions if "—" in q or " — " in q)
        assert em_dash_count <= len(questions) * 0.5, "Em dash should be used ≤50% of time"


class TestPatternDetectionHelpers:
    """Test the pattern detection helper functions."""

    def test_opener_detection(self):
        """Test opener pattern extraction."""
        assert extract_opener("That's impressive that you're building...") == "That's impressive"
        assert extract_opener("What a smart move — practicing...") == "What a smart"
        assert extract_opener("How do you approach customer discovery?") == "How [question]"
        assert extract_opener("When you think about ideal collaborators...") == "When you think"

    def test_structure_detection(self):
        """Test structure pattern detection."""
        # Em dash + bridge
        q1 = "That's impressive — building in SaaS. As you're connecting with founders..."
        assert detect_structure(q1) == "ack_em_dash_bridge_q"

        # Em dash without bridge
        q2 = "That's impressive — the combination is rare. What stage do you prefer?"
        assert detect_structure(q2) == "ack_em_dash_direct_q"

        # Bridge without em dash
        q3 = "That's interesting. Since you're building remotely, what region are you in?"
        assert detect_structure(q3) == "ack_bridge_q"

        # Parenthetical
        q4 = "What excites you most (given your background in engineering)?"
        assert detect_structure(q4) == "direct_q_parenthetical"

    def test_punctuation_detection(self):
        """Test punctuation pattern detection."""
        assert detect_punctuation_pattern("That's impressive — building tools.") == "em_dash"
        assert detect_punctuation_pattern("You mentioned AI: how do you approach it?") == "colon"
        assert detect_punctuation_pattern("What stage (pre-seed or seed)?") == "parenthetical"
        assert detect_punctuation_pattern("Building, launching, and scaling - what's first?") == "multi_comma"
        assert detect_punctuation_pattern("What's your goal?") == "simple"


@pytest.mark.integration
class TestLiveOnboardingPatternDiversity:
    """Integration tests that run actual onboarding sessions."""

    def test_full_conversation_diversity(self):
        """
        Integration test: Complete a full onboarding session and verify pattern diversity.

        This test should be run manually or in CI to catch pattern regression.
        """
        pytest.skip("Requires live API calls - run manually to verify fixes")

        # Placeholder for integration test
        # In actual implementation:
        # 1. Start onboarding session
        # 2. Send 4-5 user messages
        # 3. Collect AI questions
        # 4. Run all pattern diversity checks
        # 5. Assert PASS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
