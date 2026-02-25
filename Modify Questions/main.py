from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Any
from openai import OpenAI
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="QNA Summary Bot", version="1.0.0")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Pydantic models for request/response
class OptionItem(BaseModel):
    label: str
    value: str


class PreviousUserResponse(BaseModel):
    question_id: str
    ai_text: str
    prompt: str
    description: Optional[str] = None
    narration: Optional[str] = None
    suggestion_chips: str
    options: Optional[Union[str, List[Any]]] = None
    user_response: Optional[str] = None


class QuestionPayload(BaseModel):
    previous_user_response: Optional[List[PreviousUserResponse]] = []
    question_id: str
    code: str
    prompt: str
    narration: Optional[str] = None
    description: Optional[str] = None
    suggestion_chips: str
    options: Optional[Union[str, List[Any]]] = None


class QuestionResponse(BaseModel):
    question_id: str
    ai_text: str
    suggestion_chips: str


def format_options_for_prompt(options: Optional[Union[str, List[Any]]]) -> str:

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
            elif isinstance(item, OptionItem):
                # Handle Pydantic model
                option_labels.append(item.label)
            else:
                # Fallback to string representation
                option_labels.append(str(item))
        
        if option_labels:
            # Format as a nice list for the prompt
            return "\n".join([f"- {label}" for label in option_labels])
    
    return ""


def format_options_for_markdown(options: Optional[Union[str, List[Any]]]) -> str:

    if not options:
        return ""
    
    # If options is a string, format it as a simple text
    if isinstance(options, str):
        return f"**Options:** {options.strip()}"
    
    # If options is a list, create a Markdown list
    if isinstance(options, list):
        option_items = []
        for item in options:
            if isinstance(item, dict):
                label = item.get('label', item.get('value', str(item)))
                value = item.get('value', label)
                option_items.append(f"- **{label}**")
            elif isinstance(item, OptionItem):
                option_items.append(f"- **{item.label}**")
            else:
                option_items.append(f"- {str(item)}")
        
        if option_items:
            return "\n".join(option_items)
    
    return ""


def modify_question_tone(prompt: str, user_message: Optional[str] = None, context: Optional[str] = None, options: Optional[Union[str, List[Any]]] = None) -> str:

    # Format options for the prompt
    options_text = format_options_for_prompt(options)
    options_info = ""
    if options_text:
        options_info = f"\n\nAvailable options:\n{options_text}\n\nPlease naturally incorporate these options into the question when modifying it. Format the options nicely in Markdown."
    
    if user_message and context:
        system_prompt = """You are a helpful AI assistant that engages in natural, friendly conversations. 
        You will be given a user's previous response and a new question to ask. 
        Acknowledge the user's response naturally, then transition to asking the new question in a warm, engaging tone.
        If options are provided, include them in Markdown format (as a bulleted list or inline text).
        Return your response in Markdown format for better UI display."""
        
        user_prompt = f"""Previous conversation:
{context}

User's last response: "{user_message}"

New question to ask: {prompt}{options_info}

Please create a natural response in Markdown format that:
1. Briefly acknowledges the user's response in a friendly way
2. Smoothly transitions to asking the new question with an engaging, conversational tone
3. If options are provided, include them in Markdown format (preferably as a bulleted list)

Return the complete response in Markdown format."""
    elif user_message:
        system_prompt = """You are a helpful AI assistant that engages in natural, friendly conversations. 
        You will be given a user's response and a new question to ask. 
        Acknowledge the user's response naturally, then transition to asking the new question in a warm, engaging tone.
        If options are provided, include them in Markdown format (as a bulleted list or inline text).
        Return your response in Markdown format for better UI display."""
        
        user_prompt = f"""User's response: "{user_message}"

New question to ask: {prompt}{options_info}

Please create a natural response in Markdown format that:
1. Briefly acknowledges the user's response in a friendly way
2. Smoothly transitions to asking the new question with an engaging, conversational tone
3. If options are provided, include them in Markdown format (preferably as a bulleted list)

Return the complete response in Markdown format."""
    else:
        system_prompt = """You are a helpful AI assistant that modifies questions to have a friendly, engaging, and conversational tone. 
        If options are provided, include them in Markdown format (as a bulleted list or inline text).
        Return your response in Markdown format for better UI display."""
        
        user_prompt = f"""Please modify this question to have a friendly, engaging, and conversational tone: {prompt}{options_info}

Return the modified question in Markdown format.
If options are provided, make sure to include them naturally in Markdown format (preferably as a bulleted list) within the modified question."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


def generate_followup_question(ai_text: str, original_prompt: str) -> str:

    system_prompt = """You are a helpful AI assistant that creates engaging follow-up questions.
    Given a question that was just asked, create a related follow-up question that:
    1. Is a natural continuation or related to the original question
    2. Is concise and suitable for a suggestion chip/button
    3. Is friendly and conversational
    4. Helps guide the conversation forward
    
    Return only the follow-up question in plain text (no Markdown, no quotes, just the question)."""
    
    user_prompt = f"""The following question was just asked to a user:
{ai_text}

Original prompt context: {original_prompt}

Create a concise, friendly follow-up question that relates to this question and helps continue the conversation naturally.
Return only the question text, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        followup = response.choices[0].message.content.strip()
        # Remove any quotes if present
        followup = followup.strip('"').strip("'").strip()
        return followup
    except Exception as e:
        # Re-raise to be caught by the caller
        raise


def build_conversation_context(previous_responses: List[PreviousUserResponse]) -> str:

    if not previous_responses:
        return None
    
    context_parts = []
    for prev in previous_responses:
        # Only include responses where user_response has a value
        if prev.user_response:
            context_parts.append(f"AI: {prev.ai_text}")
            context_parts.append(f"User: {prev.user_response}")
    
    if not context_parts:
        return None
    
    return "\n".join(context_parts)


@app.post("/modify-question", response_model=QuestionResponse)
async def modify_question(payload: QuestionPayload):
   
    try:
        # Check if this is the first call
        # First call: previous_user_response array is empty OR the last item's user_response is empty/null
        is_first_call = True
        user_message = None
        context = None
        
        if payload.previous_user_response and len(payload.previous_user_response) > 0:
            # Get the last response
            last_response = payload.previous_user_response[-1]
            user_message = last_response.user_response if last_response.user_response else None
            
            # If user_response has a value, it's not the first call
            if user_message and user_message.strip():
                is_first_call = False
                # Build context from all previous responses (excluding empty ones)
                context = build_conversation_context(payload.previous_user_response)
        
        if is_first_call:
            # First time: just modify the prompt with friendly tone, including options if available
            modified_text = modify_question_tone(payload.prompt, options=payload.options)
        else:
            # Subsequent call: use user message and conversation context, including options if available
            # Use the user message and current prompt to create natural conversation flow
            modified_text = modify_question_tone(payload.prompt, user_message=user_message, context=context, options=payload.options)
        
        # Generate a follow-up question for suggestion_chips based on the modified ai_text
        try:
            suggestion_chips = generate_followup_question(modified_text, payload.prompt)
        except Exception:
            # If generation fails, fallback to original suggestion_chips from payload
            suggestion_chips = payload.suggestion_chips
        
        return QuestionResponse(
            question_id=payload.question_id,
            ai_text=modified_text,
            suggestion_chips=suggestion_chips
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "QNA Summary Bot API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


