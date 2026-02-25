# QNA Summary Bot

A FastAPI application that modifies questions with a friendly, engaging tone using OpenAI GPT-4o-mini.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. Run the application:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000
`

## API Endpoint

### POST `/modify-question`

Modifies a question's tone based on conversation context.

**Request Body:**

The same payload structure is used for both first-time and subsequent calls. The key difference is the `user_response` field in `previous_user_response`:
- **First call**: `user_response` is `null` or empty string `""`
- **Subsequent calls**: `user_response` has a value

**Example payload structure:**

```json
{
    "previous_user_response": [
        {
            "question_id": "123e4567-e89b-12d3-a456-426614174000",
            "ai_text": "What's your standout skill?",
            "prompt": "What's your standout skill?",
            "description": null,
            "narration": "Let's capture your top three strengths",
            "suggestion_chips": "\"What's your standout skill?\"",
            "options": null,
            "user_response": "I'm really good at problem-solving and coding"
        }
    ],
    "question_id": "7988fcbc-f169-4cbb-a0ae-c6bdc841558b",
    "code": "core_strength_2",
    "prompt": "What's your second strength?",
    "narration": "Let's capture your top three strengths",
    "description": null,
    "suggestion_chips": "\"What's another strength?\"",
    "options": null
}
```

**Response:**
```json
{
    "question_id": "7988fcbc-f169-4cbb-a0ae-c6bdc841558b",
    "ai_text": "Modified question in Markdown format",
    "suggestion_chips": "Your pronouns? (he/him, she/her, they/them, etc.)"
}
```

**Example with options:**
```json
{
    "previous_user_response": [],
    "question_id": "7988fcbc-f169-4cbb-a0ae-c6bdc841558b",
    "code": "gender",
    "prompt": "How do you identify?",
    "narration": null,
    "description": null,
    "suggestion_chips": "Your pronouns?",
    "options": [
        {
            "label": "Male",
            "value": "Male"
        },
        {
            "label": "Female",
            "value": "Female"
        },
        {
            "label": "Other",
            "value": "Other"
        }
    ]
}
```

## Behavior

- **First call** (when `user_response` is `null` or empty `""`): Modifies only the `prompt` field directly with a friendly, engaging tone in Markdown format.
- **Subsequent calls** (when `user_response` has a value): Uses the user's response and conversation context from previous responses to create a natural flow. Acknowledges the user's response, then asks the new question from the current `prompt` with modified tone in Markdown format.

**Note:** 
- All responses are returned in **Markdown format** for UI display (supports options, lists, formatting).
- When options are provided (as array of objects with `label` and `value`), they will be formatted nicely in Markdown (typically as a bulleted list).
- The payload structure remains the same for both first and subsequent calls.
- Only the `user_response` field value determines whether it's a first or subsequent call.
- Fields like `narration` and `description` can be `null`.

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`


