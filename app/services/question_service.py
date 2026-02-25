"""
Service for modifying questions with AI.
"""
import os
import re
import logging
from typing import List, Optional, Union, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class QuestionService:
    """Service for modifying questions with friendly, engaging tone."""
    
    def __init__(self):
        """Initialize question service."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)
        # Use environment variable or default to gpt-4.1-mini (matching Modify Questions folder)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
        # Check if model requires max_completion_tokens (o1, o3 models, or gpt-4.1-mini)
        # Some newer models require max_completion_tokens instead of max_tokens
        self.use_max_completion_tokens = self.model.startswith(('o1', 'o3')) or 'gpt-4.1-mini' in self.model
    
    def format_options_for_prompt(self, options: Optional[Union[str, List[Any]]]) -> str:
        """Format options for prompt."""
        if not options:
            return ""
        
        # If options is a string, return it as is
        if isinstance(options, str):
            return options.strip()
        
        # If options is a list, extract labels/values
        if isinstance(options, list):
            option_labels = []
            for item in options:
                if isinstance(item, dict):
                    # Handle dict format directly from JSON
                    label = item.get('label', item.get('value', str(item)))
                    option_labels.append(label)
                else:
                    # Fallback to string representation
                    option_labels.append(str(item))
            
            if option_labels:
                # Format as a nice list for the prompt
                return "\n".join([f"- {label}" for label in option_labels])
        
        return ""
    
    def modify_question_tone(
        self, 
        prompt: str, 
        user_message: Optional[str] = None, 
        context: Optional[str] = None, 
        options: Optional[Union[str, List[Any]]] = None
    ) -> str:
        """Modify question tone to be friendly and engaging."""
        # Format options for the prompt
        options_text = self.format_options_for_prompt(options)
        options_info = ""
        if options_text:
            options_info = f"\n\nAvailable options:\n{options_text}\n\nPlease naturally incorporate these options into the question when modifying it. Format the options nicely in Markdown."
        
        if user_message and context:
            system_prompt = """You are a friendly, genuine person having a soft, frank conversation. 
            You'll acknowledge what the user just said in a warm, human way, then naturally ask the next question.
            Use a soft, frank tone - be honest, straightforward, and warm. Be justifiable and reasonable.
            Be genuinely friendly and curious - like talking to a friend. Use casual, everyday language.
            If options are provided, include them naturally in Markdown format (as a bulleted list or inline text).
            Return your response in Markdown format for better UI display."""
            
            user_prompt = f"""Previous conversation:
{context}

User's last response: "{user_message}"

New question to ask: {prompt}{options_info}

Create a natural, friendly response with a soft, frank tone that:
1. Briefly acknowledges what the user said in a warm, casual way (like a friend would)
2. Smoothly transitions to asking the new question in a friendly, conversational tone
3. Use a soft, frank approach - be honest and straightforward but warm
4. Be justifiable and reasonable - explain naturally why you're asking if it makes sense
5. If options are provided, include them naturally in Markdown format (preferably as a bulleted list)
6. Sound genuinely curious and human - not robotic or formal

Return the complete response in Markdown format."""
        elif user_message:
            system_prompt = """You are a friendly, genuine person having a soft, frank conversation. 
            You'll acknowledge what the user just said in a warm, human way, then naturally ask the next question.
            Use a soft, frank tone - be honest, straightforward, and warm. Be justifiable and reasonable.
            Be genuinely friendly and curious - like talking to a friend. Use casual, everyday language.
            If options are provided, include them naturally in Markdown format (as a bulleted list or inline text).
            Return your response in Markdown format for better UI display."""
            
            user_prompt = f"""User's response: "{user_message}"

New question to ask: {prompt}{options_info}

Create a natural, friendly response with a soft, frank tone that:
1. Briefly acknowledges what the user said in a warm, casual way (like a friend would)
2. Smoothly transitions to asking the new question in a friendly, conversational tone
3. Use a soft, frank approach - be honest and straightforward but warm
4. Be justifiable and reasonable - explain naturally why you're asking if it makes sense
5. If options are provided, include them naturally in Markdown format (preferably as a bulleted list)
6. Sound genuinely curious and human - not robotic or formal

Return the complete response in Markdown format."""
        else:
            system_prompt = """You are a friendly, genuine person having a soft, frank conversation. 
            Rewrite questions in a very friendly, casual, and human way - like you're genuinely curious and asking a friend.
            Use a soft, frank tone - be honest, straightforward, and warm. Be justifiable and reasonable in your approach.
            Use natural everyday language. Avoid formal or corporate tone.
            If options are provided, include them naturally in Markdown format (as a bulleted list or inline text).
            IMPORTANT: Return ONLY the modified question text. Do NOT add any introductory phrases, explanations, or meta-commentary.
            Start directly with the question as if you're naturally asking it."""
            
            user_prompt = f"""Rewrite this question in a very friendly, casual, and human way with a soft, frank tone: {prompt}{options_info}

STYLE GUIDELINES:
- Use a soft, frank tone - be honest, straightforward, and warm
- Be justifiable and reasonable - explain why you're asking if it makes sense naturally
- Be warm, friendly, and genuinely curious
- Use casual, everyday language (like talking to a friend)
- Sound natural and human - not robotic or formal
- Be direct but gentle - frank but soft
- If options are provided, include them naturally in Markdown format (preferably as a bulleted list) within the question
- Return ONLY the question text - no introductions, no explanations, just the friendly question itself"""

        try:
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
            }
            
            # Use max_completion_tokens for o1/o3 models, max_tokens for others
            if self.use_max_completion_tokens:
                request_params["max_completion_tokens"] = 500
            else:
                request_params["max_tokens"] = 500
            
            response = self.client.chat.completions.create(**request_params)
            
            result = response.choices[0].message.content.strip()
            
            # Clean up any unwanted introductory phrases (for first call only)
            if not user_message and not context:
                result = self._clean_introductory_text(result)
            
            return result
        except Exception as e:
            error_str = str(e)
            # If we get the max_tokens error, retry with max_completion_tokens
            if "max_tokens" in error_str and "max_completion_tokens" in error_str:
                logger.warning(f"Model {self.model} requires max_completion_tokens, retrying...")
                try:
                    request_params = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.7,
                        "max_completion_tokens": 500
                    }
                    response = self.client.chat.completions.create(**request_params)
                    result = response.choices[0].message.content.strip()
                    
                    # Clean up any unwanted introductory phrases (for first call only)
                    if not user_message and not context:
                        result = self._clean_introductory_text(result)
                    
                    return result
                except Exception as retry_error:
                    logger.error(f"OpenAI API error on retry: {str(retry_error)}")
                    raise
            else:
                logger.error(f"OpenAI API error: {error_str}")
                raise
    
    def _clean_introductory_text(self, text: str) -> str:
        """Remove common introductory phrases and meta-commentary that LLM might add."""
        if not text:
            return text
        
        # Common patterns to remove (case-insensitive)
        patterns_to_remove = [
            r"^Sure!?\s*",
            r"^Here's?\s*",
            r"^Here is\s*",
            r"^Let me\s+",
            r"^I'll\s+",
            r"^I will\s+",
            r"^Of course!?\s*",
            r"^Absolutely!?\s*",
            r"^Certainly!?\s*",
            r"^a friendlier (and more engaging )?version\s*",
            r"^an engaging version\s*",
            r"^a more engaging version\s*",
        ]
        
        result = text
        
        # Remove patterns at the start
        for pattern in patterns_to_remove:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        
        # Remove separator lines (---, ===, etc.) at the start
        lines = result.split('\n')
        cleaned_lines = []
        skip_separators = True
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip separator lines at the start
            if skip_separators:
                if not line_stripped:
                    continue
                if line_stripped.replace('-', '').replace('=', '').replace('_', '').strip() == '':
                    continue
                if line_stripped.startswith(('---', '===', '___')):
                    continue
            
            # Once we have real content, stop skipping
            if line_stripped:
                skip_separators = False
            
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        
        # Remove any remaining "Here's a friendlier version:" type patterns
        result = re.sub(r"^(Here's?|Sure!?|Let me|I'll|I will)\s+[^:]*:\s*", "", result, flags=re.IGNORECASE)
        
        # Remove leading/trailing whitespace and newlines
        return result.strip()
    
    def generate_followup_question(self, ai_text: str, original_prompt: str) -> str:
        """Generate a follow-up question for suggestion chips."""
        system_prompt = """You are a friendly, genuine person creating follow-up questions with a soft, frank tone.
        Given a question that was just asked, create a related follow-up question that:
        1. Is a natural continuation or related to the original question
        2. Is concise and suitable for a suggestion chip/button
        3. Uses a soft, frank tone - honest, straightforward, and warm
        4. Is justifiable and reasonable - makes sense naturally
        5. Is friendly and conversational
        6. Helps guide the conversation forward
        
        Return only the follow-up question in plain text (no Markdown, no quotes, just the question)."""
        
        user_prompt = f"""The following question was just asked to a user:
{ai_text}

Original prompt context: {original_prompt}

Create a concise, friendly follow-up question with a soft, frank tone that:
- Relates to this question and helps continue the conversation naturally
- Uses a soft, frank approach - honest and straightforward but warm
- Is justifiable and reasonable - makes sense in context
- Is friendly and conversational

Return only the question text, nothing else."""

        try:
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
            }
            
            # Use max_completion_tokens for o1/o3 models, max_tokens for others
            if self.use_max_completion_tokens:
                request_params["max_completion_tokens"] = 500
            else:
                request_params["max_tokens"] = 500
            
            response = self.client.chat.completions.create(**request_params)
            
            followup = response.choices[0].message.content.strip()
            # Remove any quotes if present
            followup = followup.strip('"').strip("'").strip()
            return followup
        except Exception as e:
            error_str = str(e)
            # If we get the max_tokens error, retry with max_completion_tokens
            if "max_tokens" in error_str and "max_completion_tokens" in error_str:
                logger.warning(f"Model {self.model} requires max_completion_tokens, retrying...")
                try:
                    request_params = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.7,
                        "max_completion_tokens": 500
                    }
                    response = self.client.chat.completions.create(**request_params)
                    followup = response.choices[0].message.content.strip()
                    followup = followup.strip('"').strip("'").strip()
                    return followup
                except Exception as retry_error:
                    logger.error(f"Error generating follow-up question on retry: {str(retry_error)}")
                    raise
            else:
                logger.error(f"Error generating follow-up question: {error_str}")
                raise
    
    def build_conversation_context(self, previous_responses: List[Any]) -> Optional[str]:
        """Build conversation context from previous responses."""
        if not previous_responses:
            return None
        
        context_parts = []
        for prev in previous_responses:
            # Only include responses where user_response has a value
            if hasattr(prev, 'user_response') and prev.user_response:
                context_parts.append(f"AI: {prev.ai_text}")
                context_parts.append(f"User: {prev.user_response}")
        
        if not context_parts:
            return None
        
        return "\n".join(context_parts)

