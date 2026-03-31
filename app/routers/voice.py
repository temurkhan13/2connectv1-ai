"""
Voice transcription router.
Accepts audio uploads and returns text via OpenAI Whisper API.
Audio is NOT stored — transcribed and discarded immediately.
"""

import os
import logging
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice"])

# Allowed audio MIME types
ALLOWED_AUDIO_TYPES = {
    "audio/webm",
    "audio/mp4",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/aac",
    "audio/x-m4a",
    "audio/m4a",
}

MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB (Whisper API limit)


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio to text using OpenAI Whisper.

    - Accepts: webm, mp4, mpeg, wav, ogg, m4a (up to 25MB)
    - Returns: { "text": "transcribed text" }
    - Audio is discarded after transcription (not stored)
    """
    # Validate content type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {content_type}. Supported: webm, mp4, mp3, wav, ogg, m4a",
        )

    # Read audio data
    audio_data = await file.read()

    if len(audio_data) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=400, detail="Audio file too large. Maximum 25MB.")

    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Determine file extension from content type
    ext_map = {
        "audio/webm": ".webm",
        "audio/mp4": ".mp4",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/aac": ".aac",
        "audio/x-m4a": ".m4a",
        "audio/m4a": ".m4a",
    }
    ext = ext_map.get(content_type, ".webm")

    tmp_path = None
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("[VOICE] OPENAI_API_KEY not set")
            raise HTTPException(status_code=500, detail="Voice service not configured")

        client = OpenAI(api_key=api_key)

        # Write to temp file (Whisper API needs a file path with correct extension)
        tmp_fd = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_path = tmp_fd.name
        tmp_fd.write(audio_data)
        tmp_fd.close()

        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )

        logger.info(f"[VOICE] Transcribed {len(audio_data)} bytes -> {len(transcript)} chars")

        return {"text": transcript.strip()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VOICE] Transcription failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)[:100]}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
