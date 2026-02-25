"""
Resume service for handling resume processing operations.

Supports two modes:
1. URL mode: Download resume from URL (legacy)
2. Base64 mode: Process base64-encoded content from conversational upload
"""
from typing import Optional, Dict, Any, Set, Union
from urllib.parse import urlparse
import requests
import os
import tempfile
import shutil
import logging
import base64
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from app.adapters.dynamodb import UserProfile

logger = logging.getLogger(__name__)

# SECURITY: Maximum file size for resume downloads (10MB default)
MAX_RESUME_SIZE_BYTES = int(os.getenv('MAX_RESUME_SIZE_MB', '10')) * 1024 * 1024

# SECURITY: Request timeout in seconds
RESUME_DOWNLOAD_TIMEOUT = int(os.getenv('RESUME_DOWNLOAD_TIMEOUT', '30'))

# SECURITY: Allowed domains for direct resume downloads (SSRF protection)
# If empty, only backend-proxied downloads are allowed
ALLOWED_RESUME_DOMAINS: Set[str] = set(
    filter(None, os.getenv('ALLOWED_RESUME_DOMAINS', '').split(','))
)


class ResumeService:
    """Service for resume processing operations."""

    def __init__(self):
        """Initialize resume service."""
        pass

    def _is_url_allowed(self, url: str) -> bool:
        """
        Check if URL is from an allowed domain (SSRF protection).

        Returns True if:
        - URL uses backend proxy (always allowed)
        - URL domain is in allowlist
        - Allowlist is empty AND local_mode is set (development)
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]

            # Check against allowlist
            if ALLOWED_RESUME_DOMAINS:
                return domain in ALLOWED_RESUME_DOMAINS

            # If no allowlist configured, only allow in local mode
            local_mode = os.getenv("LOCAL_MODE", "")
            if local_mode:
                logger.warning(f"No ALLOWED_RESUME_DOMAINS configured - allowing {domain} in local mode")
                return True

            return False
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            return False

    def _download_with_size_limit(self, url: str, headers: Dict[str, str] = None) -> bytes:
        """
        Download file with size limit and timeout (DoS protection).

        Raises:
            ValueError: If file exceeds size limit
            requests.RequestException: If download fails
        """
        response = requests.get(
            url,
            headers=headers or {},
            stream=True,
            timeout=RESUME_DOWNLOAD_TIMEOUT
        )
        response.raise_for_status()

        # Check Content-Length header first (if available)
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > MAX_RESUME_SIZE_BYTES:
            raise ValueError(f"Resume file too large: {int(content_length) / 1024 / 1024:.1f}MB exceeds {MAX_RESUME_SIZE_BYTES / 1024 / 1024:.0f}MB limit")

        # Stream download with size enforcement
        chunks = []
        total_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            total_size += len(chunk)
            if total_size > MAX_RESUME_SIZE_BYTES:
                raise ValueError(f"Resume file exceeds {MAX_RESUME_SIZE_BYTES / 1024 / 1024:.0f}MB size limit")
            chunks.append(chunk)

        return b''.join(chunks)
    
    def _cleanup_text(self, text: str) -> str:
        """Clean up extracted text by normalizing whitespace and removing excessive blank lines."""
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        return text.strip()
    
    def _get_file_extension(self, url: str) -> str:
        """Extract file extension from URL."""
        return os.path.splitext(url.split('?')[0])[1].lower()
    
    def _save_upload_to_temp(self, content: bytes, suffix: str) -> str:
        """Save content to temporary file and return path."""
        fd, path = tempfile.mkstemp(suffix=suffix or "")
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        return path

    def _process_base64_resume(self, user_id: str, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process base64-encoded resume content from conversational upload.

        Args:
            user_id: User identifier
            resume_data: Dict with:
                - content: Base64-encoded file content
                - filename: Original filename
                - content_type: MIME type
        """
        logger.info(f"Processing base64 resume for user {user_id}: {resume_data.get('filename', 'unknown')}")

        try:
            # Decode base64 content
            content_b64 = resume_data.get('content', '')
            content = base64.b64decode(content_b64)
            filename = resume_data.get('filename', 'resume.pdf')
            content_type = resume_data.get('content_type', 'application/pdf')

            # Validate size
            if len(content) > MAX_RESUME_SIZE_BYTES:
                return {
                    "success": False,
                    "skipped": False,
                    "reason": "File too large",
                    "message": f"Resume exceeds {MAX_RESUME_SIZE_BYTES / 1024 / 1024:.0f}MB size limit"
                }

            # Determine file extension from content type or filename
            if 'pdf' in content_type.lower():
                suffix = '.pdf'
            elif 'word' in content_type.lower() or filename.endswith('.docx'):
                suffix = '.docx'
            elif filename.endswith('.doc'):
                suffix = '.doc'
            else:
                # Fallback to extension from filename
                suffix = os.path.splitext(filename)[1].lower() or '.pdf'

            # Get/create user profile
            try:
                user_profile = UserProfile.get(user_id)
            except UserProfile.DoesNotExist:
                logger.warning(f"User profile {user_id} not found for resume processing")
                return {
                    "success": False,
                    "skipped": False,
                    "reason": "User profile not found",
                    "message": f"User profile {user_id} not found"
                }

            user_profile.processing_status = 'processing'
            user_profile.save()

            # Save to temp file and extract text
            temp_path = self._save_upload_to_temp(content, suffix)
            try:
                text = ""
                if suffix == '.pdf':
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    text = "\n\n".join(d.page_content for d in docs)
                    extraction_method = "PyPDFLoader"
                elif suffix == '.docx':
                    loader = Docx2txtLoader(temp_path)
                    docs = loader.load()
                    text = "\n\n".join(d.page_content for d in docs)
                    extraction_method = "Docx2txtLoader"
                elif suffix in ('.txt', '.text'):
                    loader = TextLoader(temp_path, encoding="utf-8")
                    docs = loader.load()
                    text = "\n\n".join(d.page_content for d in docs)
                    extraction_method = "TextLoader"
                else:
                    raise ValueError(f"Unsupported file type: {suffix}")

                # Clean up the text
                text = self._cleanup_text(text)

                if not text:
                    raise ValueError("No text extracted from the resume")

                # Store extracted text in DynamoDB
                user_profile.resume_text.text = text
                user_profile.resume_text.extracted_at = datetime.utcnow()
                user_profile.resume_text.extraction_method = f"{extraction_method} (conversational_upload)"
                user_profile.processing_status = 'completed'
                user_profile.persona_status = 'pending'
                user_profile.save()

                logger.info(f"Successfully processed resume for user {user_id}. Extracted {len(text)} characters.")

                return {
                    "success": True,
                    "skipped": False,
                    "reason": "Resume processed successfully",
                    "message": f"Resume processed - extracted {len(text)} characters from {filename}"
                }

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except base64.binascii.Error as e:
            logger.error(f"Invalid base64 content for user {user_id}: {e}")
            return {
                "success": False,
                "skipped": False,
                "reason": "Invalid encoding",
                "message": "Resume content is not valid base64"
            }
        except ValueError as e:
            logger.error(f"Resume parsing failed for user {user_id}: {e}")
            try:
                user_profile = UserProfile.get(user_id)
                user_profile.processing_status = 'failed_parsing'
                user_profile.save()
            except Exception:
                pass
            return {
                "success": False,
                "skipped": False,
                "reason": "Parsing failed",
                "message": f"Resume parsing failed: {str(e)}"
            }
        except Exception as e:
            logger.exception(f"Unexpected error processing base64 resume for {user_id}: {e}")
            try:
                user_profile = UserProfile.get(user_id)
                user_profile.processing_status = 'failed_unknown'
                user_profile.save()
            except Exception:
                pass
            return {
                "success": False,
                "skipped": False,
                "reason": "Unexpected error",
                "message": f"Unexpected error: {str(e)}"
            }

    def process_resume(self, user_id: str, resume_data: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Process resume and extract text.

        Args:
            user_id: User identifier
            resume_data: One of:
                - None: Skip processing
                - str: URL to download resume from (legacy)
                - dict: Base64-encoded content from conversational upload with keys:
                    - content: Base64-encoded file content
                    - filename: Original filename
                    - content_type: MIME type
        """
        # SECURITY: Never log the full resume URL (could contain tokens)
        logger.info(f"Starting resume processing for user: {user_id}")

        if not resume_data:
            logger.info(f"No resume data provided for user {user_id}. Skipping resume processing.")
            # Update user profile to indicate no resume was processed
            try:
                user_profile = UserProfile.get(user_id)
                user_profile.processing_status = 'completed'
                user_profile.resume_text = {"text": "", "extraction_method": "none", "extracted_at": datetime.now()}
                user_profile.save()
                logger.info(f"User profile updated for {user_id} - no resume processing needed")
                return {
                    "success": True,
                    "skipped": True,
                    "reason": "No resume data provided",
                    "message": "Resume processing skipped - no resume data provided"
                }
            except Exception as e:
                logger.error(f"Error updating user profile for {user_id}: {str(e)}")
                return {
                    "success": False,
                    "skipped": False,
                    "reason": "Database error",
                    "message": f"Error updating user profile: {str(e)}"
                }

        # Handle base64-encoded content (from conversational onboarding upload)
        if isinstance(resume_data, dict) and 'content' in resume_data:
            return self._process_base64_resume(user_id, resume_data)

        # Handle URL (legacy)
        resume_link = resume_data if isinstance(resume_data, str) else None
        if not resume_link:
            logger.warning(f"Invalid resume_data format for user {user_id}")
            return {
                "success": False,
                "skipped": False,
                "reason": "Invalid format",
                "message": "Resume data must be a URL string or dict with 'content' key"
            }

        try:
            user_profile = UserProfile.get(user_id)
            user_profile.processing_status = 'processing'
            user_profile.save()

            # Download the resume with SSRF protection
            logger.info(f"Downloading resume for user {user_id}...")
            local_mode = os.getenv("LOCAL_MODE", "")
            backend_url = os.getenv("RECIPROCITY_BACKEND_URL", "")

            if backend_url and not local_mode:
                # Use backend proxy for downloads (safest option)
                webhook_key = os.getenv("WEBHOOK_API_KEY", "")
                stream_url = f"{backend_url}/api/v1/webhooks/stream-file"
                headers = {"x-api-key": webhook_key} if webhook_key else {}
                # Backend handles URL validation
                content = self._download_with_size_limit(
                    f"{stream_url}?url={requests.utils.quote(resume_link, safe='')}",
                    headers
                )
            else:
                # Direct download - check domain allowlist (SSRF protection)
                if not self._is_url_allowed(resume_link):
                    logger.warning(f"Resume URL domain not in allowlist for user {user_id}")
                    return {
                        "success": False,
                        "skipped": False,
                        "reason": "URL not allowed",
                        "message": "Resume URL domain not in allowed list"
                    }
                content = self._download_with_size_limit(resume_link)

            # Determine file type and save to temp file
            suffix = self._get_file_extension(resume_link)
            temp_path = self._save_upload_to_temp(content, suffix)
            
            try:
                # Extract text using appropriate LangChain loader
                logger.info(f"Extracting text from resume for user {user_id}...")
                text = ""

                if suffix in (".pdf",):
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    text = "\n\n".join(d.page_content for d in docs)
                    extraction_method = "PyPDFLoader"
                elif suffix in (".docx",):
                    loader = Docx2txtLoader(temp_path)
                    docs = loader.load()
                    text = "\n\n".join(d.page_content for d in docs)
                    extraction_method = "Docx2txtLoader"
                elif suffix in (".txt", ".text"):
                    loader = TextLoader(temp_path, encoding="utf-8")
                    docs = loader.load()
                    text = "\n\n".join(d.page_content for d in docs)
                    extraction_method = "TextLoader"
                else:
                    raise ValueError(f"Unsupported file type: {suffix}")

                # Clean up the text
                text = self._cleanup_text(text)

                if not text:
                    raise ValueError("No text extracted from the resume.")

                # Store extracted text in DynamoDB
                user_profile.resume_text.text = text
                user_profile.resume_text.extracted_at = datetime.utcnow()
                user_profile.resume_text.extraction_method = extraction_method
                user_profile.processing_status = 'completed'
                user_profile.persona_status = 'pending'
                user_profile.save()

                logger.info(f"Successfully processed resume for user {user_id}. Extracted {len(text)} characters.")
                
                return {
                    "success": True,
                    "skipped": False,
                    "reason": "Resume processed successfully",
                    "message": f"Resume processed successfully - extracted {len(text)} characters"
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
        except UserProfile.DoesNotExist:
            logger.warning(f"User profile {user_id} not found for resume processing.")
            return {
                "success": False,
                "skipped": False,
                "reason": "User profile not found",
                "message": f"User profile {user_id} not found for resume processing"
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download resume for user {user_id}: {e}")
            try:
                user_profile = UserProfile.get(user_id)
                user_profile.processing_status = 'failed_download'
                user_profile.save()
            except Exception:
                pass
            return {
                "success": False,
                "skipped": False,
                "reason": "Download failed",
                "message": f"Failed to download resume: {str(e)}"
            }
        except ValueError as e:
            logger.error(f"Resume parsing failed for user {user_id}: {e}")
            try:
                user_profile = UserProfile.get(user_id)
                user_profile.processing_status = 'failed_parsing'
                user_profile.save()
            except Exception:
                pass
            return {
                "success": False,
                "skipped": False,
                "reason": "Parsing failed",
                "message": f"Resume parsing failed: {str(e)}"
            }
        except Exception as e:
            logger.exception(f"An unexpected error occurred during resume processing for user {user_id}: {e}")
            try:
                user_profile = UserProfile.get(user_id)
                user_profile.processing_status = 'failed_unknown'
                user_profile.save()
            except Exception:
                pass
            return {
                "success": False,
                "skipped": False,
                "reason": "Unexpected error",
                "message": f"Unexpected error: {str(e)}"
            }
