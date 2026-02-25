"""
AI Chat Service for simulating AI-to-AI conversations between users.
Enhanced with goal-focused prompts, misalignment detection, and dynamic responses.
"""
import os
import random
import json
import openai
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from app.adapters.dynamodb import UserProfile

logger = logging.getLogger(__name__)


class AIChatService:
    """Service for managing AI-to-AI chat conversations."""
    
    def __init__(self):
        """Initialize AI chat service."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
    
    def get_persona(self, user_id: str) -> Dict[str, Any]:
        """
        Fetch persona data from DynamoDB.
        """
        try:
            user = UserProfile.get(user_id)
            if user and user.persona:
                persona_dict = {}
                if hasattr(user.persona, 'name'):
                    persona_dict['name'] = user.persona.name or f"User-{user_id[:8]}"
                else:
                    persona_dict['name'] = f"User-{user_id[:8]}"
                    
                if hasattr(user.persona, 'designation'):
                    persona_dict['designation'] = user.persona.designation or "Professional"
                else:
                    persona_dict['designation'] = "Professional"
                    
                if hasattr(user.persona, 'archetype'):
                    persona_dict['archetype'] = user.persona.archetype or ""
                else:
                    persona_dict['archetype'] = ""
                    
                if hasattr(user.persona, 'focus'):
                    persona_dict['focus'] = user.persona.focus or ""
                else:
                    persona_dict['focus'] = ""
                    
                if hasattr(user.persona, 'requirements'):
                    persona_dict['requirements'] = user.persona.requirements or ""
                else:
                    persona_dict['requirements'] = ""
                    
                if hasattr(user.persona, 'offerings'):
                    persona_dict['offerings'] = user.persona.offerings or ""
                else:
                    persona_dict['offerings'] = ""
                
                return persona_dict
        except UserProfile.DoesNotExist:
            logger.warning(f"User profile not found for {user_id}")
        except Exception as e:
            logger.error(f"Error fetching persona for user {user_id}: {str(e)}")
        
        return {
            "name": f"User-{user_id[:8]}",
            "designation": "Professional",
            "archetype": "",
            "focus": "",
            "requirements": "",
            "offerings": ""
        }
    
    def build_system_prompt(self, persona: Dict[str, Any], other_persona: Dict[str, Any], role: str, stage: str, alignment_info: Dict[str, Any] = None, is_first_message: bool = False) -> str:
        """
        Build a single, concise system prompt for AI-to-AI chat.
        Enforces strict Initiator (Requirements) / Responder (Offerings) structure.
        """
        # Extract details
        my_name = persona.get('name', 'Unknown')
        my_role = persona.get('designation', 'Professional')
        my_seeking = persona.get('requirements', 'Not specified')
        my_offering = persona.get('offerings', 'Not specified')
        
        their_name = other_persona.get('name', 'Unknown')
        their_role = other_persona.get('designation', 'Professional')
        their_seeking = other_persona.get('requirements', 'Not specified')
        their_offering = other_persona.get('offerings', 'Not specified')

        # Determine strict goal based on role
        if role.lower() == "initiator" and stage == "introduction" and is_first_message:
            # Initiator MUST state requirements first
            current_instruction = f"You are the INITIATOR. Start with a polite greeting. Then, state clearly what you are SEEKING. Summarize your requirements naturally: '{my_seeking}'. Be very concise (1 sentence preferred)."
        elif role.lower() == "responder" and stage == "introduction" and is_first_message:
            # Responder MUST state offerings in response
            current_instruction = f"You are the RESPONDER. Start with 'Hello' or 'Hi'. State what you can OFFER. Summarize your offering naturally: '{my_offering}'. Be very concise (1 sentence preferred). Do NOT copy-paste the offering text."
        else:
            # Continue naturally based on alignment
            alignment_type = alignment_info.get('alignment_type', 'unknown') if alignment_info else 'unknown'
            if alignment_type == "misaligned":
                current_instruction = "The goals do NOT align. Politely acknowledge the mismatch and end the conversation."
            elif alignment_type == "fully_aligned":
                current_instruction = "The goals align well. Express enthusiasm and suggest connecting offline. End the conversation."
            else:
                current_instruction = "There is some potential match. Suggest exchanging contact details to explore further. End the conversation."

        # Single, concise prompt
        prompt = f"""You are {my_name}, a {my_role}.
Your GOAL is to find: {my_seeking}.
You can OFFER: {my_offering}.

You are talking to {their_name}, a {their_role}.
They are looking for: {their_seeking}.
They can offer: {their_offering}.

INSTRUCTION: {current_instruction}

STYLE:
- Natural, human, and polite.
- You MUST start with 'Hello' or 'Hi' if it is YOUR first message.
- Do NOT use greetings (Hello/Hi) in subsequent messages.
- Very short messages (1 sentence preferred, 2 max).
- No filler words.
- Summarize long text into natural speech. Do NOT copy-paste persona fields.
- Do NOT invent specific dates or times (like 'Wednesday at 2 PM').
- Do NOT share fake or real email addresses, phone numbers, or links.
- Do NOT ask 'When are you available?'. Suggest connecting offline.
- Do NOT mention or invent specific third parties (like 'she', 'he', 'my colleague'). Speak only for YOURSELF.
- Use straight quotes (') instead of smart quotes (’).
- If goals don't match, say so and say goodbye.
"""
        return prompt
    
    def check_alignment(self, persona_a: Dict[str, Any], persona_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check initial alignment between two personas using GPT.
        Returns alignment score and analysis.
        """
        if not self.client:
            return {"alignment_score": 50, "alignment_type": "partially_aligned", "reason": "Unable to analyze", "should_continue_chat": True}
        
        try:
            prompt = f"""Analyze if these two people can ACTUALLY help each other:

PERSON A ({persona_a.get('name', 'Unknown')}):
- Role: {persona_a.get('designation', 'Unknown')}
- SEEKING: {persona_a.get('requirements', 'Not specified')}
- OFFERING: {persona_a.get('offerings', 'Not specified')}

PERSON B ({persona_b.get('name', 'Unknown')}):
- Role: {persona_b.get('designation', 'Unknown')}
- SEEKING: {persona_b.get('requirements', 'Not specified')}
- OFFERING: {persona_b.get('offerings', 'Not specified')}

=== STRICT MATCHING TEST ===

ASK THESE QUESTIONS:
Q1: Can A give what B needs? (Does A's OFFERING match B's SEEKING?)
Q2: Can B give what A needs? (Does B's OFFERING match A's SEEKING?)

If BOTH answers are NO → MISALIGNED (Score 15-30)
If ONE answer is YES → PARTIALLY ALIGNED (Score 40-60)
If BOTH answers are YES → FULLY ALIGNED (Score 70-85)

=== MISALIGNED EXAMPLES (Score 15-30) ===
- Engineer seeking CTO role + Investor seeking LPs = MISALIGNED (neither can help!)
- Founder seeking funding + Mentor seeking mentees = MISALIGNED
- Job seeker + Job seeker = MISALIGNED
- Two investors both seeking startups = MISALIGNED

=== ALIGNED EXAMPLES (Score 70-85) ===
- Startup seeking funding + Investor seeking startups = ALIGNED
- Junior seeking mentor + Mentor seeking mentees = ALIGNED
- Company needing developer + Developer seeking job = ALIGNED

IMPORTANT: Same industry/sector is NOT enough! They must be able to HELP each other.

Respond ONLY in JSON:
{{"alignment_score": <15-85>, "alignment_type": "<fully_aligned|partially_aligned|misaligned>", "reason": "<Can A help B? Can B help A?>", "should_continue_chat": <false if misaligned>, "common_ground": "<specific match or 'none'>"}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an alignment analyzer. Respond ONLY in valid JSON, no other text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            # Clean up potential markdown formatting
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            logger.info(f"Alignment check result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error checking alignment: {str(e)}")
            return {"alignment_score": 50, "alignment_type": "partially_aligned", "reason": "Analysis error", "should_continue_chat": True, "common_ground": "unknown"}
    
    def get_dynamic_max_tokens(self, stage: str, message_index: int, alignment_type: str = "partially_aligned") -> int:
        """
        Return max_tokens ensuring COMPLETE sentences without cutoffs.
        Balanced for brevity (1-2 sentences).
        """
        if alignment_type == "misaligned":
            base_tokens = {
                "introduction": (60, 100),
                "discovery": (60, 100),
                "closing": (50, 80)
            }
        elif alignment_type == "fully_aligned":
            base_tokens = {
                "introduction": (70, 110),
                "discovery": (70, 110),
                "closing": (60, 90)
            }
        else:
            base_tokens = {
                "introduction": (70, 110),
                "discovery": (70, 110),
                "evaluation": (70, 110),
                "deep_dive": (70, 110),
                "closing": (60, 90)
            }
        
        min_tokens, max_tokens = base_tokens.get(stage, (80, 120))
        
        # Small variation
        variation = random.randint(-5, 5)
        tokens = random.randint(min_tokens, max_tokens) + variation
        
        return max(60, min(150, tokens))
    
    def is_too_similar(self, new_response: str, previous_messages: List[Dict[str, str]], threshold: float = 0.5) -> bool:
        """
        Check if new response is too similar to any previous message.
        Lower threshold (0.5) = more strict about repetition.
        """
        if not previous_messages:
            return False
        
        new_lower = new_response.lower()
        new_words = set(new_lower.split())
        
        # Check for repeated introduction patterns
        intro_patterns = ["hi, i'm", "hello, i'm", "i'm a", "i am a", "my name is"]
        has_intro = any(p in new_lower for p in intro_patterns)
        
        for msg in previous_messages:
            prev_lower = msg.get('content', '').lower()
            prev_words = set(prev_lower.split())
            
            # If new message has intro and previous also had intro = REPEAT
            if has_intro and any(p in prev_lower for p in intro_patterns):
                return True
            
            if not prev_words:
                continue
            
            # Calculate word overlap
            overlap = len(new_words & prev_words)
            total = len(new_words | prev_words)
            
            if total > 0 and (overlap / total) > threshold:
                return True
        
        return False
    
    def generate_ai_response(
        self,
        conversation_history: List[Dict[str, str]],
        persona: Dict[str, Any],
        other_persona: Dict[str, Any],
        role: str,
        stage: str,
        alignment_info: Dict[str, Any] = None
    ) -> str:
        """
        Generate AI response using OpenAI with goal-focused prompts.
        Includes anti-repetition check.
        """
        if not self.client:
            return f"Hello, I'm {persona.get('name', 'Unknown')}. Nice to meet you!"
        
        # Determine if this is the first message for this persona
        # If history is empty, it's Initiator's first message
        # If history has 1 message, it's Responder's first message
        is_first_message = len(conversation_history) <= 1
        
        system_prompt = self.build_system_prompt(persona, other_persona, role, stage, alignment_info, is_first_message)
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Dynamic token count for varied message lengths based on alignment
        alignment_type = alignment_info.get('alignment_type', 'partially_aligned') if alignment_info else 'partially_aligned'
        max_tokens = self.get_dynamic_max_tokens(stage, len(conversation_history), alignment_type)
        
        try:
            # Try up to 3 times to get a non-repetitive response
            for attempt in range(3):
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0
                )
                response_text = response.choices[0].message.content.strip()
                
                # Check if response is too similar to previous messages
                if not self.is_too_similar(response_text, conversation_history):
                    return response_text
                
                logger.warning(f"Response too similar on attempt {attempt + 1}, retrying...")
            
            # If all attempts fail, return the last response anyway
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return f"I appreciate the conversation. Let me think about what you've shared."
    
    def _build_history(self, conversation_data: List[Dict[str, str]], current_role_id: str) -> List[Dict[str, str]]:
        """
        Build conversation history from the perspective of the current role (assistant).
        """
        history = []
        for msg in conversation_data:
            role = "assistant" if msg["sender_id"] == current_role_id else "user"
            history.append({
                "role": role,
                "content": msg["content"]
            })
        return history

    def simulate_conversation(
        self,
        initiator_id: str,
        responder_id: str,
        match_id: str,
        template: str = None
    ) -> Dict[str, Any]:
        """
        Simulate full AI-to-AI conversation with alignment detection and early close.
        """
        logger.info(f"Starting AI chat simulation between {initiator_id} and {responder_id}")
        
        # Fetch personas
        initiator_persona = self.get_persona(initiator_id)
        responder_persona = self.get_persona(responder_id)
        
        logger.info(f"Initiator persona: {initiator_persona.get('name')} - {initiator_persona.get('designation')}")
        logger.info(f"Responder persona: {responder_persona.get('name')} - {responder_persona.get('designation')}")
        
        # Check initial alignment
        alignment_info = self.check_alignment(initiator_persona, responder_persona)
        logger.info(f"Alignment analysis: {alignment_info.get('alignment_type')} - Score: {alignment_info.get('alignment_score')}")
        
        # Initialize conversation
        conversation_data = []
        
        # Determine conversation flow based on alignment - 3 LEVELS
        alignment_type = alignment_info.get('alignment_type', 'partially_aligned')
        alignment_score = alignment_info.get('alignment_score', 50)
        
        if alignment_type == 'misaligned' or alignment_score < 30:
            # LEVEL 3: Completely Opposite - VERY SHORT conversation, early close
            # Only 2-3 exchanges then polite rejection
            stages = ["introduction", "closing"]
            logger.info(f"LEVEL 3: Completely Opposite (score: {alignment_score}) - minimal conversation, early close")
        elif alignment_type == 'partially_aligned' or (alignment_score >= 30 and alignment_score < 70):
            # LEVEL 2: Partially Aligned - medium conversation
            stages = ["introduction", "discovery", "evaluation", "closing"]
            logger.info(f"LEVEL 2: Partially Aligned (score: {alignment_score}) - medium conversation")
        else:
            # LEVEL 1: Fully Aligned - shorter but meaningful conversation (6 messages)
            stages = ["introduction", "discovery", "closing"]
            logger.info(f"LEVEL 1: Fully Aligned (score: {alignment_score}) - concise conversation")
        
        # Generate initial message from initiator
        if template:
            initiator_message = template
        else:
            initiator_message = self.generate_ai_response(
                [],
                initiator_persona,
                responder_persona,
                "initiator",
                "introduction",
                alignment_info
            )
        
        # Add initiator's opening message
        conversation_data.append({
            "sender_id": initiator_id,
            "content": initiator_message
        })
        
        # Simulate conversation through stages
        early_close = False
        
        # For MISALIGNED - 4-6 messages (2-3 per side) so both goals are clear
        if alignment_type == 'misaligned' or alignment_score < 30:
            logger.info("MISALIGNED: 4-6 message exchange (2-3 per side)")
            
            # Define sequence of turns
            steps = [
                ("responder", "introduction"),
                ("initiator", "discovery"),
                ("responder", "closing"),
                ("initiator", "closing")
            ]
            
            for role, stage in steps:
                current_id = initiator_id if role == "initiator" else responder_id
                current_persona = initiator_persona if role == "initiator" else responder_persona
                other_persona = responder_persona if role == "initiator" else initiator_persona
                
                history = self._build_history(conversation_data, current_id)
                message = self.generate_ai_response(
                    history,
                    current_persona,
                    other_persona,
                    role,
                    stage,
                    alignment_info
                )
                conversation_data.append({
                    "sender_id": current_id,
                    "content": message
                })
                
                # Check for goodbye/rejection to stop early
                if "goodbye" in message.lower() or "best of luck" in message.lower():
                    break
            
            early_close = True
        
        # For ALIGNED/PARTIAL - normal flow
        for stage_idx, stage in enumerate(stages):
            if early_close:
                break
                
            # Responder replies
            history = self._build_history(conversation_data, responder_id)
            responder_message = self.generate_ai_response(
                history,
                responder_persona,
                initiator_persona,
                "responder",
                stage,
                alignment_info
            )
            
            conversation_data.append({
                "sender_id": responder_id,
                "content": responder_message
            })
            
            # Check for rejection in conversation - close early if found
            rejection_indicators = [
                "don't think i can assist", "not a good fit", "goals don't align",
                "can't help you", "not what i'm looking for", "outside my scope",
                "paths may not align", "not the right match", "goodbye"
            ]
            if any(ind in responder_message.lower() for ind in rejection_indicators):
                logger.info("Rejection detected - closing conversation")
                early_close = True
                break
            
            # If not closing stage and not early close, initiator responds
            if stage != "closing" and not early_close:
                history = self._build_history(conversation_data, initiator_id)
                initiator_message = self.generate_ai_response(
                    history,
                    initiator_persona,
                    responder_persona,
                    "initiator",
                    stage,
                    alignment_info
                )
                
                conversation_data.append({
                    "sender_id": initiator_id,
                    "content": initiator_message
                })
            
            if early_close:
                # Check if last message already has closing words to avoid double goodbye
                last_msg = conversation_data[-1]['content'].lower()
                closing_words = ["goodbye", "bye", "best of luck", "take care", "wish you the best"]
                if any(word in last_msg for word in closing_words):
                    break
                
                # Add final graceful close from initiator
                history = self._build_history(conversation_data, initiator_id)
                close_message = self.generate_ai_response(
                    history,
                    initiator_persona,
                    responder_persona,
                    "initiator",
                    "closing",
                    alignment_info
                )
                conversation_data.append({
                    "sender_id": initiator_id,
                    "content": close_message
                })
                break
        
        # Generate AI-based compatibility analysis
        compatibility_result = self.calculate_compatibility_score(
            initiator_persona,
            responder_persona,
            conversation_data,
            alignment_info
        )
        
        # Generate conversation summary with conclusion
        ai_remarks = self.generate_conversation_summary(
            conversation_data,
            initiator_persona,
            responder_persona,
            alignment_info,
            compatibility_result
        )
        
        return {
            "initiator_id": initiator_id,
            "responder_id": responder_id,
            "match_id": match_id,
            "conversation_data": conversation_data,
            "ai_remarks": ai_remarks,
            "compatibility_score": compatibility_result['score'],
            "alignment_type": alignment_info.get('alignment_type', 'unknown'),
            "early_close": early_close
        }
    
    def generate_conversation_summary(
        self,
        conversation_data: List[Dict[str, str]],
        initiator_persona: Dict[str, Any],
        responder_persona: Dict[str, Any],
        alignment_info: Dict[str, Any],
        compatibility_result: Dict[str, Any]
    ) -> str:
        """
        Generate AI remarks based on honest conversation analysis.
        Verdict is determined by actual alignment, not forced.
        """
        alignment_type = alignment_info.get('alignment_type', 'unknown')
        score = compatibility_result.get('score', 50)
        
        if not self.client:
            if score < 40 or alignment_type == 'misaligned':
                return f"VERDICT: REJECTED\nREASON: No mutual benefit found between {initiator_persona['name']} and {responder_persona['name']}.\nMUTUAL BENEFIT: No\nCONFIDENCE: High"
            elif score >= 70 and alignment_type == 'fully_aligned':
                return f"VERDICT: ACCEPTED\nREASON: Strong alignment found.\nMUTUAL BENEFIT: Yes\nCONFIDENCE: High"
            return f"VERDICT: NEEDS REVIEW\nREASON: Partial alignment.\nMUTUAL BENEFIT: Partial\nCONFIDENCE: Medium"
        
        try:
            conversation_text = "\n".join([
                f"{msg['sender_id'][:8]}: {msg['content']}"
                for msg in conversation_data
            ])
            
            # Determine verdict based on score first
            if score >= 60:
                score_verdict = "ACCEPTED"
            elif score >= 40:
                score_verdict = "NEEDS_REVIEW"
            else:
                score_verdict = "REJECTED"
            
            # Determine conclusion based on level
            if score >= 65:
                level = "LEVEL 1 - STRONG MATCH"
                conclusion = "These parties should connect. High potential for collaboration."
            elif score >= 40:
                level = "LEVEL 2 - PARTIAL MATCH"
                conclusion = "Worth exploring further. Some alignment exists but needs validation."
            else:
                level = "LEVEL 3 - NO MATCH"
                conclusion = "Not recommended. Goals do not align. Both parties seeking different outcomes."
            
            prompt = f"""Provide a CLEAR CONCLUSION for this match.

PERSON A: {initiator_persona.get('name')} ({initiator_persona.get('designation')})
- SEEKING: {initiator_persona.get('requirements', 'Not specified')}
- OFFERING: {initiator_persona.get('offerings', 'Not specified')}

PERSON B: {responder_persona.get('name')} ({responder_persona.get('designation')})
- SEEKING: {responder_persona.get('requirements', 'Not specified')}
- OFFERING: {responder_persona.get('offerings', 'Not specified')}

CLASSIFICATION: {level}
Compatibility Score: {score}

CONVERSATION:
{conversation_text[:1200]}

Write a summary with CLEAR CONCLUSION:

**VERDICT: {score_verdict}**

**CLASSIFICATION:** {level}

**REASON:** [1-2 sentences explaining WHY this score - what aligns or doesn't align]

**MUTUAL BENEFIT:** [Yes/Partial/No] - [Brief explanation]

**CONCLUSION:** {conclusion}

Keep it concise and professional. The verdict is already decided based on the score."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a match summary writer. The verdict is already decided based on the score. Your job is to explain WHY the match has this score and provide supporting analysis. Do not contradict the given verdict."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            if alignment_type == 'misaligned' or score < 40:
                return "VERDICT: REJECTED\nREASON: Analysis indicates misaligned goals between parties.\nMUTUAL BENEFIT: No"
            return "VERDICT: NEEDS REVIEW\nREASON: Please manually review conversation."
    
    def calculate_compatibility_score(
        self,
        initiator_persona: Dict[str, Any],
        responder_persona: Dict[str, Any],
        conversation_data: List[Dict[str, str]],
        alignment_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate compatibility score using AI analysis instead of heuristics.
        """
        if not self.client:
            # Fallback to alignment score if available
            if alignment_info:
                return {
                    "score": alignment_info.get('alignment_score', 50),
                    "confidence": "low",
                    "reason": "Fallback score from initial alignment"
                }
            return {"score": 50, "confidence": "low", "reason": "Unable to analyze"}
        
        try:
            conversation_text = "\n".join([
                f"{msg['content']}" for msg in conversation_data
            ])
            
            prompt = f"""Calculate compatibility score based on whether they can HELP each other:

PERSON A:
- SEEKING: {initiator_persona.get('requirements', 'Not specified')}
- OFFERING: {initiator_persona.get('offerings', 'Not specified')}

PERSON B:
- SEEKING: {responder_persona.get('requirements', 'Not specified')}
- OFFERING: {responder_persona.get('offerings', 'Not specified')}

=== STRICT SCORING ===

STEP 1: Can A provide what B needs? YES/NO
STEP 2: Can B provide what A needs? YES/NO

SCORING:
- Both NO → Score 15-25 (NO MATCH - neither can help!)
- One YES → Score 40-55 (PARTIAL - one-way benefit)
- Both YES → Score 70-85 (STRONG - mutual benefit)

EXAMPLES OF NO MATCH (15-25):
- Engineer seeking CTO role + Investor seeking LP capital = 20 (neither helps!)
- Job seeker + Job seeker = 15
- Investor + Investor = 20

EXAMPLES OF MATCH (70-85):
- Startup seeking funding + Investor seeking startups = 80
- Mentee seeking guidance + Mentor seeking mentees = 80

CONVERSATION CHECK:
{conversation_text[:600]}

Did they acknowledge mismatch? → Lower score
Did they find real alignment? → Higher score

Respond ONLY in JSON:
{{"score": <15-85>, "confidence": "<high|medium|low>", "reason": "<Can they help each other? YES/NO>"}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a compatibility analyzer. Two people seeking the same thing = NO MATCH (0-25). But when one OFFERS what the other SEEKS, that's a MATCH (76-100). Mentor seeking mentees + mentee seeking mentor = FULL MATCH because both achieve their goals. Investor seeking startups + startup seeking funding = FULL MATCH. Respond ONLY in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            # Clean up potential markdown
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            
            # Cap score to avoid extremes (15-90 range)
            raw_score = result.get('score', 50)
            if raw_score < 15:
                result['score'] = 15
            elif raw_score > 90:
                result['score'] = 90
            
            logger.info(f"Compatibility analysis: Score={result.get('score')}, Confidence={result.get('confidence')}")
            return result
        except Exception as e:
            logger.error(f"Error calculating compatibility score: {str(e)}")
            # Fallback to alignment info
            if alignment_info:
                return {
                    "score": alignment_info.get('alignment_score', 50),
                    "confidence": "low",
                    "reason": "Fallback to initial alignment"
                }
            return {"score": 50, "confidence": "low", "reason": "Analysis error"}
