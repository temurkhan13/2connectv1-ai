"""
AI Conversation Enhancement Service.

Provides intelligent conversation support for matched users,
including context-aware suggestions, conversation guidance,
and compatibility insights during chat.

Key features:
1. Context-aware conversation suggestions
2. Real-time compatibility insights
3. Topic suggestions based on profiles
4. Conversation health monitoring
5. Meeting readiness assessment
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConversationStage(str, Enum):
    """Stage of the conversation between matched users."""
    INITIAL = "initial"              # First messages
    EXPLORING = "exploring"          # Learning about each other
    DEEPENING = "deepening"          # Discussing specifics
    EVALUATING = "evaluating"        # Assessing fit
    READY_TO_MEET = "ready_to_meet"  # Ready for call/meeting
    STALLED = "stalled"              # Conversation not progressing


class SuggestionType(str, Enum):
    """Types of conversation suggestions."""
    TOPIC = "topic"                  # Topic to discuss
    QUESTION = "question"            # Question to ask
    SHARE = "share"                  # Something to share about yourself
    CLARIFY = "clarify"              # Clarification needed
    NEXT_STEP = "next_step"          # Suggest moving forward
    RE_ENGAGE = "re_engage"          # Re-engage stalled conversation


class ConversationHealth(str, Enum):
    """Health status of conversation."""
    HEALTHY = "healthy"              # Good flow, balanced
    ONE_SIDED = "one_sided"          # One person doing most talking
    SURFACE_LEVEL = "surface_level"  # Not getting deep enough
    STALLED = "stalled"              # No recent activity
    READY_TO_ADVANCE = "ready_to_advance"  # Ready to move forward


@dataclass
class ConversationMessage:
    """A message in the conversation."""
    message_id: str
    sender_id: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSuggestion:
    """A suggestion for the conversation."""
    suggestion_id: str
    suggestion_type: SuggestionType
    content: str
    reasoning: str
    relevance_score: float
    based_on: List[str]  # What context was used


@dataclass
class ConversationInsight:
    """Insight about the conversation."""
    insight_type: str
    title: str
    description: str
    action_suggested: Optional[str] = None


@dataclass
class ConversationAnalysis:
    """Full analysis of a conversation."""
    conversation_id: str
    stage: ConversationStage
    health: ConversationHealth
    message_count: int
    turn_balance: float  # 0-1, 0.5 is perfectly balanced
    topics_discussed: List[str]
    topics_to_explore: List[str]
    suggestions: List[ConversationSuggestion]
    insights: List[ConversationInsight]
    meeting_readiness: float  # 0-1
    analyzed_at: datetime


class AIConversationEnhancer:
    """
    Enhances conversations between matched users.

    Provides real-time suggestions, monitors conversation health,
    and helps users have more productive conversations.
    """

    # Topic suggestions by persona type
    TOPIC_BANKS = {
        "investor_founder": [
            "Investment thesis alignment",
            "Fundraising timeline and goals",
            "Traction and key metrics",
            "Team composition and backgrounds",
            "Market opportunity and competition",
            "Use of funds and milestones",
            "Board dynamics and governance",
            "Exit expectations and timeline"
        ],
        "investor_investor": [
            "Investment focus and thesis",
            "Portfolio synergies",
            "Co-investment opportunities",
            "Market views and trends",
            "Deal flow sharing",
            "LP relationships",
            "Fund strategy evolution"
        ],
        "founder_founder": [
            "Scaling challenges",
            "Hiring and team building",
            "Fundraising experiences",
            "Customer acquisition strategies",
            "Product development approaches",
            "Work-life balance",
            "Mentorship and advisors"
        ],
        "general": [
            "Current projects and priorities",
            "Industry trends and observations",
            "Challenges you're facing",
            "Goals for the next year",
            "How you got started",
            "What excites you most"
        ]
    }

    # Question templates
    QUESTION_TEMPLATES = {
        "initial": [
            "What drew you to {their_focus}?",
            "How did you get started in {their_industry}?",
            "What's your current focus right now?"
        ],
        "exploring": [
            "What's the biggest challenge you're facing in {topic}?",
            "How do you approach {topic}?",
            "What's worked well for you in {topic}?"
        ],
        "deepening": [
            "Can you tell me more about your experience with {specific_topic}?",
            "What would an ideal outcome look like for you?",
            "What are your key criteria for {their_goal}?"
        ],
        "evaluating": [
            "What would make this a valuable connection for you?",
            "Are there specific ways we could work together?",
            "What questions do you have for me?"
        ]
    }

    def __init__(self):
        # Conversation tracking (in production, use database)
        self._conversations: Dict[str, List[ConversationMessage]] = defaultdict(list)
        self._analyses: Dict[str, ConversationAnalysis] = {}

        # Configuration
        self.stall_threshold_hours = int(os.getenv("CONVERSATION_STALL_HOURS", "48"))
        self.meeting_ready_threshold = float(os.getenv("MEETING_READY_THRESHOLD", "0.7"))

    def add_message(
        self,
        conversation_id: str,
        sender_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> ConversationMessage:
        """
        Add a message to the conversation.

        Args:
            conversation_id: Conversation identifier
            sender_id: Sender's user ID
            content: Message content
            metadata: Optional metadata

        Returns:
            ConversationMessage record
        """
        message = ConversationMessage(
            message_id=f"msg_{datetime.utcnow().timestamp()}",
            sender_id=sender_id,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

        self._conversations[conversation_id].append(message)
        logger.debug(f"Added message to conversation {conversation_id}")

        return message

    def analyze_conversation(
        self,
        conversation_id: str,
        viewer_persona: Dict[str, Any],
        other_persona: Dict[str, Any]
    ) -> ConversationAnalysis:
        """
        Analyze conversation and generate insights.

        Args:
            conversation_id: Conversation identifier
            viewer_persona: Persona of the viewing user
            other_persona: Persona of the other user

        Returns:
            ConversationAnalysis with stage, health, and suggestions
        """
        messages = self._conversations.get(conversation_id, [])

        # Determine stage
        stage = self._determine_stage(messages)

        # Assess health
        health, turn_balance = self._assess_health(messages, viewer_persona.get("user_id"))

        # Extract discussed topics
        topics_discussed = self._extract_topics(messages)

        # Identify topics to explore
        topics_to_explore = self._suggest_topics(
            viewer_persona, other_persona, topics_discussed
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(
            stage, health, viewer_persona, other_persona, messages
        )

        # Generate insights
        insights = self._generate_insights(
            stage, health, turn_balance, topics_discussed
        )

        # Assess meeting readiness
        meeting_readiness = self._assess_meeting_readiness(
            stage, health, len(messages), topics_discussed
        )

        analysis = ConversationAnalysis(
            conversation_id=conversation_id,
            stage=stage,
            health=health,
            message_count=len(messages),
            turn_balance=turn_balance,
            topics_discussed=topics_discussed,
            topics_to_explore=topics_to_explore,
            suggestions=suggestions,
            insights=insights,
            meeting_readiness=meeting_readiness,
            analyzed_at=datetime.utcnow()
        )

        self._analyses[conversation_id] = analysis
        return analysis

    def _determine_stage(self, messages: List[ConversationMessage]) -> ConversationStage:
        """Determine conversation stage based on messages."""
        if not messages:
            return ConversationStage.INITIAL

        count = len(messages)
        last_message_age = datetime.utcnow() - messages[-1].timestamp

        # Check for stalled
        if last_message_age > timedelta(hours=self.stall_threshold_hours):
            return ConversationStage.STALLED

        # Stage by message count (simplified heuristic)
        if count <= 4:
            return ConversationStage.INITIAL
        elif count <= 10:
            return ConversationStage.EXPLORING
        elif count <= 20:
            return ConversationStage.DEEPENING
        else:
            # Check content for meeting-related keywords
            recent_content = " ".join(m.content.lower() for m in messages[-5:])
            meeting_keywords = ["call", "meet", "schedule", "calendar", "zoom", "coffee"]
            if any(kw in recent_content for kw in meeting_keywords):
                return ConversationStage.READY_TO_MEET
            return ConversationStage.EVALUATING

    def _assess_health(
        self,
        messages: List[ConversationMessage],
        viewer_id: str
    ) -> Tuple[ConversationHealth, float]:
        """Assess conversation health and balance."""
        if not messages:
            return ConversationHealth.HEALTHY, 0.5

        # Calculate turn balance
        viewer_messages = sum(1 for m in messages if m.sender_id == viewer_id)
        total = len(messages)
        balance = viewer_messages / total if total > 0 else 0.5

        # Check for stalled
        last_message_age = datetime.utcnow() - messages[-1].timestamp
        if last_message_age > timedelta(hours=self.stall_threshold_hours):
            return ConversationHealth.STALLED, balance

        # Check for one-sided
        if balance < 0.3 or balance > 0.7:
            return ConversationHealth.ONE_SIDED, balance

        # Check for surface level (short messages)
        avg_length = sum(len(m.content) for m in messages) / total
        if avg_length < 50 and total > 6:
            return ConversationHealth.SURFACE_LEVEL, balance

        # Check if ready to advance
        if total >= 15 and 0.4 <= balance <= 0.6:
            return ConversationHealth.READY_TO_ADVANCE, balance

        return ConversationHealth.HEALTHY, balance

    def _extract_topics(self, messages: List[ConversationMessage]) -> List[str]:
        """Extract topics discussed from messages."""
        topics = []
        all_content = " ".join(m.content.lower() for m in messages)

        # Topic keywords to detect
        topic_keywords = {
            "investment": ["invest", "funding", "raise", "round", "capital"],
            "product": ["product", "feature", "build", "develop", "ship"],
            "market": ["market", "customer", "user", "growth", "traction"],
            "team": ["team", "hire", "culture", "people", "talent"],
            "strategy": ["strategy", "plan", "roadmap", "vision", "goal"],
            "competition": ["competitor", "market share", "differentiate"],
            "metrics": ["metric", "kpi", "revenue", "arr", "mrr", "growth rate"],
            "experience": ["experience", "background", "previous", "worked at"]
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in all_content for kw in keywords):
                topics.append(topic)

        return topics

    def _suggest_topics(
        self,
        viewer: Dict[str, Any],
        other: Dict[str, Any],
        already_discussed: List[str]
    ) -> List[str]:
        """Suggest topics to explore."""
        # Determine conversation type
        viewer_type = self._get_user_type(viewer)
        other_type = self._get_user_type(other)

        conv_type = f"{viewer_type}_{other_type}"
        if conv_type not in self.TOPIC_BANKS:
            conv_type = "general"

        # Get relevant topics not yet discussed
        all_topics = self.TOPIC_BANKS[conv_type]
        suggestions = []

        for topic in all_topics:
            topic_lower = topic.lower()
            # Check if related topics already discussed
            is_discussed = any(
                discussed in topic_lower or topic_lower in discussed
                for discussed in already_discussed
            )
            if not is_discussed:
                suggestions.append(topic)

        return suggestions[:5]

    def _get_user_type(self, persona: Dict[str, Any]) -> str:
        """Determine user type from persona."""
        archetype = persona.get("archetype", "").lower()
        if "investor" in archetype or "angel" in archetype or "vc" in archetype:
            return "investor"
        elif "founder" in archetype or "ceo" in archetype or "entrepreneur" in archetype:
            return "founder"
        return "general"

    def _generate_suggestions(
        self,
        stage: ConversationStage,
        health: ConversationHealth,
        viewer: Dict[str, Any],
        other: Dict[str, Any],
        messages: List[ConversationMessage]
    ) -> List[ConversationSuggestion]:
        """Generate conversation suggestions."""
        suggestions = []

        # Stage-specific suggestions
        if stage == ConversationStage.INITIAL:
            suggestions.append(ConversationSuggestion(
                suggestion_id=f"sug_{datetime.utcnow().timestamp()}_1",
                suggestion_type=SuggestionType.QUESTION,
                content=f"Ask about their experience in {other.get('focus', 'their field')[:30]}",
                reasoning="Opening with genuine curiosity builds rapport",
                relevance_score=0.9,
                based_on=["stage", "other_persona"]
            ))

        elif stage == ConversationStage.EXPLORING:
            suggestions.append(ConversationSuggestion(
                suggestion_id=f"sug_{datetime.utcnow().timestamp()}_2",
                suggestion_type=SuggestionType.SHARE,
                content="Share a specific challenge you're working through",
                reasoning="Vulnerability builds trust and invites deeper discussion",
                relevance_score=0.85,
                based_on=["stage"]
            ))

        elif stage == ConversationStage.STALLED:
            suggestions.append(ConversationSuggestion(
                suggestion_id=f"sug_{datetime.utcnow().timestamp()}_3",
                suggestion_type=SuggestionType.RE_ENGAGE,
                content="Share an interesting update or ask about their recent progress",
                reasoning="Re-engage with something new rather than following up emptily",
                relevance_score=0.95,
                based_on=["stage", "health"]
            ))

        # Health-specific suggestions
        if health == ConversationHealth.ONE_SIDED:
            suggestions.append(ConversationSuggestion(
                suggestion_id=f"sug_{datetime.utcnow().timestamp()}_4",
                suggestion_type=SuggestionType.QUESTION,
                content="Ask an open-ended question to hear more from them",
                reasoning="Balance the conversation by encouraging their input",
                relevance_score=0.88,
                based_on=["health"]
            ))

        elif health == ConversationHealth.SURFACE_LEVEL:
            suggestions.append(ConversationSuggestion(
                suggestion_id=f"sug_{datetime.utcnow().timestamp()}_5",
                suggestion_type=SuggestionType.TOPIC,
                content="Go deeper on a topic - share a specific story or ask 'why'",
                reasoning="Moving past surface level builds meaningful connection",
                relevance_score=0.87,
                based_on=["health"]
            ))

        elif health == ConversationHealth.READY_TO_ADVANCE:
            suggestions.append(ConversationSuggestion(
                suggestion_id=f"sug_{datetime.utcnow().timestamp()}_6",
                suggestion_type=SuggestionType.NEXT_STEP,
                content="Consider suggesting a video call to continue the conversation",
                reasoning="Conversation has good momentum - time to deepen the connection",
                relevance_score=0.92,
                based_on=["health", "stage"]
            ))

        return suggestions[:4]

    def _generate_insights(
        self,
        stage: ConversationStage,
        health: ConversationHealth,
        balance: float,
        topics: List[str]
    ) -> List[ConversationInsight]:
        """Generate insights about the conversation."""
        insights = []

        # Balance insight
        if balance < 0.4:
            insights.append(ConversationInsight(
                insight_type="balance",
                title="They're doing most of the talking",
                description="Consider sharing more about yourself to balance the conversation",
                action_suggested="Share your perspective on the topics discussed"
            ))
        elif balance > 0.6:
            insights.append(ConversationInsight(
                insight_type="balance",
                title="You're doing most of the talking",
                description="Try asking more questions to learn about them",
                action_suggested="Ask open-ended questions about their experience"
            ))

        # Topics insight
        if len(topics) >= 4:
            insights.append(ConversationInsight(
                insight_type="progress",
                title="Good topic coverage",
                description=f"You've discussed {len(topics)} different areas",
                action_suggested=None
            ))
        elif len(topics) <= 1 and stage not in [ConversationStage.INITIAL]:
            insights.append(ConversationInsight(
                insight_type="depth",
                title="Conversation staying narrow",
                description="Consider exploring other relevant topics",
                action_suggested="Branch into related areas of interest"
            ))

        # Stage insight
        if stage == ConversationStage.READY_TO_MEET:
            insights.append(ConversationInsight(
                insight_type="milestone",
                title="Ready for next step",
                description="The conversation suggests you're ready for a call or meeting",
                action_suggested="Propose a time to connect live"
            ))

        return insights

    def _assess_meeting_readiness(
        self,
        stage: ConversationStage,
        health: ConversationHealth,
        message_count: int,
        topics: List[str]
    ) -> float:
        """Assess readiness for moving to a meeting."""
        score = 0.0

        # Stage contribution
        stage_scores = {
            ConversationStage.INITIAL: 0.1,
            ConversationStage.EXPLORING: 0.3,
            ConversationStage.DEEPENING: 0.6,
            ConversationStage.EVALUATING: 0.8,
            ConversationStage.READY_TO_MEET: 0.95,
            ConversationStage.STALLED: 0.2
        }
        score += stage_scores.get(stage, 0.3) * 0.4

        # Health contribution
        if health == ConversationHealth.HEALTHY:
            score += 0.25
        elif health == ConversationHealth.READY_TO_ADVANCE:
            score += 0.3
        elif health == ConversationHealth.ONE_SIDED:
            score += 0.1
        elif health == ConversationHealth.STALLED:
            score += 0.05

        # Message count contribution
        if message_count >= 20:
            score += 0.2
        elif message_count >= 10:
            score += 0.15
        elif message_count >= 5:
            score += 0.1

        # Topics contribution
        if len(topics) >= 4:
            score += 0.15
        elif len(topics) >= 2:
            score += 0.1

        return min(1.0, score)

    def get_real_time_suggestion(
        self,
        conversation_id: str,
        viewer_id: str,
        last_message: str,
        viewer_persona: Dict[str, Any],
        other_persona: Dict[str, Any]
    ) -> Optional[ConversationSuggestion]:
        """
        Get a real-time suggestion based on the last message.

        Args:
            conversation_id: Conversation identifier
            viewer_id: Viewer's user ID
            last_message: Content of the last message
            viewer_persona: Viewer's persona
            other_persona: Other user's persona

        Returns:
            ConversationSuggestion or None
        """
        last_message_lower = last_message.lower()

        # Detect question - suggest answering with depth
        if "?" in last_message:
            return ConversationSuggestion(
                suggestion_id=f"rt_{datetime.utcnow().timestamp()}",
                suggestion_type=SuggestionType.SHARE,
                content="They asked a question - consider giving a detailed, thoughtful answer",
                reasoning="Detailed answers show engagement and build connection",
                relevance_score=0.85,
                based_on=["last_message"]
            )

        # Detect short message - suggest engaging more
        if len(last_message) < 30:
            return ConversationSuggestion(
                suggestion_id=f"rt_{datetime.utcnow().timestamp()}",
                suggestion_type=SuggestionType.QUESTION,
                content="Their message was brief - ask a follow-up question to keep momentum",
                reasoning="Short messages can signal disengagement - re-engage with curiosity",
                relevance_score=0.7,
                based_on=["last_message"]
            )

        # Detect meeting mention
        meeting_words = ["call", "meet", "schedule", "zoom", "coffee", "chat live"]
        if any(word in last_message_lower for word in meeting_words):
            return ConversationSuggestion(
                suggestion_id=f"rt_{datetime.utcnow().timestamp()}",
                suggestion_type=SuggestionType.NEXT_STEP,
                content="They mentioned meeting - respond with specific availability",
                reasoning="Strike while interest is high - propose concrete times",
                relevance_score=0.95,
                based_on=["last_message"]
            )

        return None

    def analysis_to_dict(self, analysis: ConversationAnalysis) -> Dict[str, Any]:
        """Convert analysis to dictionary for API."""
        return {
            "conversation_id": analysis.conversation_id,
            "stage": analysis.stage.value,
            "health": analysis.health.value,
            "message_count": analysis.message_count,
            "turn_balance": round(analysis.turn_balance, 2),
            "topics_discussed": analysis.topics_discussed,
            "topics_to_explore": analysis.topics_to_explore,
            "suggestions": [
                {
                    "id": s.suggestion_id,
                    "type": s.suggestion_type.value,
                    "content": s.content,
                    "reasoning": s.reasoning,
                    "relevance": round(s.relevance_score, 2)
                }
                for s in analysis.suggestions
            ],
            "insights": [
                {
                    "type": i.insight_type,
                    "title": i.title,
                    "description": i.description,
                    "action": i.action_suggested
                }
                for i in analysis.insights
            ],
            "meeting_readiness": round(analysis.meeting_readiness, 2),
            "analyzed_at": analysis.analyzed_at.isoformat()
        }


# Global instance
ai_conversation_enhancer = AIConversationEnhancer()
