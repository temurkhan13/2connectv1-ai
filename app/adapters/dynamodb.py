"""DynamoDB adapter for user profile persistence."""
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from pynamodb.attributes import UnicodeAttribute, MapAttribute, NumberAttribute, UTCDateTimeAttribute, ListAttribute
from pynamodb.models import Model

load_dotenv()

logger = logging.getLogger(__name__)


class QuestionAnswer(MapAttribute):
    """PynamoDB map attribute for question-answer pairs."""
    prompt = UnicodeAttribute()
    answer = UnicodeAttribute()


class ProfileData(MapAttribute):
    """PynamoDB map attribute for user profile data."""
    resume_link = UnicodeAttribute(null=True)
    raw_questions = ListAttribute(of=QuestionAnswer, default=list)
    created_at = UTCDateTimeAttribute()
    updated_at = UTCDateTimeAttribute()


class ResumeTextData(MapAttribute):
    """PynamoDB map attribute for resume text data."""
    text = UnicodeAttribute(null=True)
    extracted_at = UTCDateTimeAttribute(null=True)
    extraction_method = UnicodeAttribute(null=True)


class PersonaData(MapAttribute):
    """PynamoDB map attribute for persona data."""
    name = UnicodeAttribute(null=True)
    archetype = UnicodeAttribute(null=True)
    designation = UnicodeAttribute(null=True)
    experience = UnicodeAttribute(null=True)
    focus = UnicodeAttribute(null=True)
    profile_essence = UnicodeAttribute(null=True)
    investment_philosophy = UnicodeAttribute(null=True)  # Deprecated: use 'strategy'
    strategy = UnicodeAttribute(null=True)  # Role-agnostic strategy field
    what_theyre_looking_for = UnicodeAttribute(null=True)
    engagement_style = UnicodeAttribute(null=True)
    requirements = UnicodeAttribute(null=True)
    offerings = UnicodeAttribute(null=True)
    # Added for match explanations
    user_type = UnicodeAttribute(null=True)
    industry = UnicodeAttribute(null=True)
    generated_at = UTCDateTimeAttribute(null=True)


class UserProfile(Model):
    """PynamoDB model for user profiles."""

    class Meta:
        table_name = os.getenv('DYNAMO_PROFILE_TABLE_NAME')
        # Support both AWS_DEFAULT_REGION and AWS_REGION (fallback)
        region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION', 'us-east-1')
        # Only set host for local development (LocalStack)
        # Support both DYNAMODB_ENDPOINT_URL and AWS_ENDPOINT_URL
        host = os.getenv('DYNAMODB_ENDPOINT_URL') or os.getenv('AWS_ENDPOINT_URL')
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        # AWS SDK handles region and credential detection automatically
        # Only validate table name since we need to know which table to use
        if not table_name:
            raise ValueError("DYNAMO_PROFILE_TABLE_NAME environment variable is required")
    
    user_id = UnicodeAttribute(hash_key=True)
    profile = ProfileData()
    resume_text = ResumeTextData()
    persona = PersonaData()
    processing_status = UnicodeAttribute(default='not_initiated')
    persona_status = UnicodeAttribute(default='not_initiated')
    needs_matchmaking = UnicodeAttribute(default='true')

    @classmethod
    def create_user(cls, user_id: str, resume_link: Optional[str], questions: List[Dict[str, Any]]) -> 'UserProfile':
        """Create a new user profile."""
        now = datetime.utcnow()
        profile_data = ProfileData(
            resume_link=resume_link,
            raw_questions=[],
            created_at=now,
            updated_at=now
        )
        
        for q_data in questions:
            profile_data.raw_questions.append(
                QuestionAnswer(prompt=q_data['prompt'], answer=str(q_data['answer']))
            )

        return cls(
            user_id=user_id,
            profile=profile_data,
            resume_text=ResumeTextData(text=None, extracted_at=None, extraction_method=None),
            persona=PersonaData(
                name=None, 
                archetype=None, 
                designation=None,  
                experience=None,
                focus=None,
                profile_essence=None,
                investment_philosophy=None,
                what_theyre_looking_for=None,
                engagement_style=None,
                requirements=None,
                offerings=None,
                generated_at=None
            ),
            processing_status='pending',
            persona_status='not_initiated'
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert user profile to dictionary."""
        return {
            "user_id": self.user_id,
            "profile": {
                "resume_link": self.profile.resume_link,
                "raw_questions": [q.as_dict() for q in self.profile.raw_questions],
                "created_at": self.profile.created_at.isoformat() if self.profile.created_at else None,
                "updated_at": self.profile.updated_at.isoformat() if self.profile.updated_at else None,
            },
            "resume_text": {
                "text": self.resume_text.text,
                "extracted_at": self.resume_text.extracted_at.isoformat() if self.resume_text.extracted_at else None,
                "extraction_method": self.resume_text.extraction_method,
            } if self.resume_text.text else None,
            "persona": {
                "name": self.persona.name,
                "archetype": self.persona.archetype,
                "designation": self.persona.designation,  
                "experience": self.persona.experience,
                "focus": self.persona.focus,
                "profile_essence": self.persona.profile_essence,
                "investment_philosophy": self.persona.investment_philosophy,
                "what_theyre_looking_for": self.persona.what_theyre_looking_for,
                "engagement_style": self.persona.engagement_style,
                "requirements": self.persona.requirements,
                "offerings": self.persona.offerings,
                "generated_at": self.persona.generated_at.isoformat() if self.persona.generated_at else None,
            } if self.persona else None,
            "processing_status": self.processing_status,
            "persona_status": self.persona_status,
            "needs_matchmaking": self.needs_matchmaking
        }


class MatchRecord(MapAttribute):
    """Individual match record"""
    matched_user_id = UnicodeAttribute()
    similarity_score = NumberAttribute()
    match_type = UnicodeAttribute()
    explanation = UnicodeAttribute(null=True)
    created_at = UTCDateTimeAttribute(default=datetime.utcnow)


class UserMatches(Model):
    """Store user matches persistently"""

    class Meta:
        table_name = os.getenv('DYNAMO_MATCHES_TABLE_NAME', 'user_matches')
        region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION', 'us-east-1')
        host = os.getenv('DYNAMODB_ENDPOINT_URL') or os.getenv('AWS_ENDPOINT_URL')
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    user_id = UnicodeAttribute(hash_key=True)
    matches = ListAttribute(of=MatchRecord, default=list)
    total_matches = NumberAttribute(default=0)
    last_updated = UTCDateTimeAttribute(default=datetime.utcnow)
    
    @classmethod
    def store_user_matches(cls, user_id: str, matches_data: dict) -> bool:
        """Store matches for a user"""
        if not user_id:
            logger.warning("store_user_matches called with empty user_id")
            return False
        try:
            # Get existing or create new
            try:
                user_matches = cls.get(user_id)
            except cls.DoesNotExist:
                user_matches = cls(user_id=user_id)
            
            # Clear existing matches
            user_matches.matches = []
            
            # Add requirements matches
            for match in matches_data.get('requirements_matches', []):
                match_record = MatchRecord(
                    matched_user_id=match['user_id'],
                    similarity_score=match['similarity_score'],
                    match_type='requirements',
                    explanation=match.get('explanation', ''),
                    created_at=datetime.utcnow()
                )
                user_matches.matches.append(match_record)
            
            # Add offerings matches  
            for match in matches_data.get('offerings_matches', []):
                match_record = MatchRecord(
                    matched_user_id=match['user_id'],
                    similarity_score=match['similarity_score'], 
                    match_type='offerings',
                    explanation=match.get('explanation', ''),
                    created_at=datetime.utcnow()
                )
                user_matches.matches.append(match_record)
            
            # Update metadata
            user_matches.total_matches = len(user_matches.matches)
            user_matches.last_updated = datetime.utcnow()
            
            # Save to DynamoDB
            user_matches.save()
            
            logger.info(f"Stored {user_matches.total_matches} matches for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing matches for user {user_id}: {e}")
            return False
    
    @classmethod
    def get_user_matches(cls, user_id: str) -> Optional[Dict[str, Any]]:
        """Get stored matches for a user"""
        if not user_id:
            logger.warning("get_user_matches called with empty user_id")
            return None
        try:
            user_matches = cls.get(user_id)
            
            requirements_matches = []
            offerings_matches = []
            
            for match in user_matches.matches:
                match_data = {
                    'user_id': match.matched_user_id,
                    'similarity_score': float(match.similarity_score),
                    'match_type': match.match_type,
                    'explanation': match.explanation,
                    'created_at': match.created_at.isoformat() if match.created_at else None
                }
                
                if match.match_type == 'requirements':
                    requirements_matches.append(match_data)
                else:
                    offerings_matches.append(match_data)
            
            return {
                'user_id': user_id,
                'total_matches': int(user_matches.total_matches),
                'requirements_matches': requirements_matches,
                'offerings_matches': offerings_matches,
                'last_updated': user_matches.last_updated.isoformat() if user_matches.last_updated else None
            }
            
        except cls.DoesNotExist:
            logger.debug(f"No stored matches found for user {user_id}")
            return {
                'user_id': user_id,
                'total_matches': 0,
                'requirements_matches': [],
                'offerings_matches': [],
                'last_updated': None
            }
        except Exception as e:
            logger.error(f"Error getting matches for user {user_id}: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'total_matches': int(self.total_matches),
            'matches': [
                {
                    'matched_user_id': match.matched_user_id,
                    'similarity_score': float(match.similarity_score),
                    'match_type': match.match_type,
                    'explanation': match.explanation,
                    'created_at': match.created_at.isoformat() if match.created_at else None
                } for match in self.matches
            ],
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
    
    @classmethod
    def clear_user_matches(cls, user_id: str) -> bool:
        """
        Clear all stored matches for a user (used when profile updates).
        This allows fresh matching without old data.
        """
        if not user_id:
            logger.warning("clear_user_matches called with empty user_id")
            return False
        try:
            try:
                user_matches = cls.get(user_id)
                user_matches.delete()
                logger.info(f"Cleared all matches for user {user_id}")
                return True
            except cls.DoesNotExist:
                logger.debug(f"No matches to clear for user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error clearing matches for user {user_id}: {e}")
            return False


class Feedback(Model):
    """
    PynamoDB model for storing user feedback on matches/chats.
    """

    class Meta:
        table_name = os.getenv('DYNAMO_FEEDBACK_TABLE_NAME', 'user_feedback')
        region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION', 'us-east-1')
        host = os.getenv('DYNAMODB_ENDPOINT_URL') or os.getenv('AWS_ENDPOINT_URL')
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    feedback_id = UnicodeAttribute(hash_key=True)
    user_id = UnicodeAttribute()
    type = UnicodeAttribute()  # "match" or "chat"
    target_id = UnicodeAttribute()  # match_id or chat_id being reviewed
    feedback = UnicodeAttribute()
    created_at = UTCDateTimeAttribute(default=datetime.utcnow)


class NotifiedMatchPairs(Model):
    """
    Track which match pairs have already been notified.
    Prevents duplicate/repeat notifications for same pair.
    pair_key = sorted(user_a, user_b) joined - ensures A-B == B-A
    """

    class Meta:
        table_name = os.getenv('DYNAMO_NOTIFIED_PAIRS_TABLE_NAME', 'notified_match_pairs')
        region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION', 'us-east-1')
        host = os.getenv('DYNAMODB_ENDPOINT_URL') or os.getenv('AWS_ENDPOINT_URL')
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    pair_key = UnicodeAttribute(hash_key=True)  # "userA_userB" sorted
    user_a_id = UnicodeAttribute()
    user_b_id = UnicodeAttribute()
    notified_at = UTCDateTimeAttribute(default=datetime.utcnow)
    notification_count = NumberAttribute(default=1)  # how many times notified
    last_similarity_score = NumberAttribute(null=True)
    
    @staticmethod
    def generate_pair_key(user_id_1: str, user_id_2: str) -> Optional[str]:
        """Generate consistent pair key regardless of order (A-B == B-A)"""
        if not user_id_1 or not user_id_2:
            logger.warning(f"generate_pair_key called with invalid IDs: {user_id_1}, {user_id_2}")
            return None
        sorted_ids = sorted([user_id_1, user_id_2])
        return f"{sorted_ids[0]}_{sorted_ids[1]}"
    
    @classmethod
    def is_pair_notified(cls, user_id_1: str, user_id_2: str) -> bool:
        """Check if this pair has already been notified before"""
        try:
            pair_key = cls.generate_pair_key(user_id_1, user_id_2)
            if not pair_key:
                return False
            cls.get(pair_key)
            return True
        except cls.DoesNotExist:
            return False
        except Exception as e:
            logger.error(f"Error checking notified pair: {e}")
            return False
    
    @classmethod
    def mark_pair_notified(cls, user_id_1: str, user_id_2: str, similarity_score: float = None) -> bool:
        """Mark a pair as notified (or increment count if already exists)"""
        try:
            pair_key = cls.generate_pair_key(user_id_1, user_id_2)
            if not pair_key:
                return False
            sorted_ids = sorted([user_id_1, user_id_2])
            
            try:
                # If pair already exists, increment notification count
                existing = cls.get(pair_key)
                existing.notification_count = (existing.notification_count or 1) + 1
                existing.notified_at = datetime.utcnow()
                if similarity_score:
                    existing.last_similarity_score = similarity_score
                existing.save()
            except cls.DoesNotExist:
                # New pair, create record
                new_pair = cls(
                    pair_key=pair_key,
                    user_a_id=sorted_ids[0],
                    user_b_id=sorted_ids[1],
                    notified_at=datetime.utcnow(),
                    notification_count=1,
                    last_similarity_score=similarity_score
                )
                new_pair.save()
            
            return True
        except Exception as e:
            logger.error(f"Error marking pair notified: {e}")
            return False
    
    @classmethod
    def get_user_notified_count(cls, user_id: str) -> int:
        """Get how many unique pairs this user has been notified about"""
        try:
            count = 0
            # Scan for pairs containing this user
            for pair in cls.scan():
                if pair.user_a_id == user_id or pair.user_b_id == user_id:
                    count += 1
            return count
        except Exception as e:
            logger.error(f"Error getting user notified count: {e}")
            return 0
    
    @classmethod
    def clear_user_pairs(cls, user_id: str) -> int:
        """
        Clear all notified pairs involving this user (used when profile updates).
        This allows the user to be re-matched and re-notified with fresh matches.
        
        Returns:
            Number of pairs deleted
        """
        try:
            deleted_count = 0
            pairs_to_delete = []
            
            # Find all pairs containing this user
            for pair in cls.scan():
                if pair.user_a_id == user_id or pair.user_b_id == user_id:
                    pairs_to_delete.append(pair)
            
            # Delete each pair
            for pair in pairs_to_delete:
                try:
                    pair.delete()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting pair {pair.pair_key}: {e}")
            
            logger.info(f"Cleared {deleted_count} notified pairs for user {user_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error clearing notified pairs for user {user_id}: {e}")
            return 0


class ChatMessage(MapAttribute):
    """PynamoDB map attribute for chat messages."""
    sender_id = UnicodeAttribute()
    content = UnicodeAttribute()
    timestamp = UTCDateTimeAttribute(default=datetime.utcnow)


class ChatRecord(Model):
    """
    PynamoDB model for storing AI-to-AI chat conversations.
    """

    class Meta:
        table_name = os.getenv('DYNAMO_CHAT_TABLE_NAME', 'ai_chat_records')
        region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION', 'us-east-1')
        host = os.getenv('DYNAMODB_ENDPOINT_URL') or os.getenv('AWS_ENDPOINT_URL')
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    chat_id = UnicodeAttribute(hash_key=True)
    match_id = UnicodeAttribute()
    initiator_id = UnicodeAttribute()
    responder_id = UnicodeAttribute()
    conversation_data = ListAttribute(of=ChatMessage)
    ai_remarks = UnicodeAttribute(null=True)
    compatibility_score = NumberAttribute(null=True)
    created_at = UTCDateTimeAttribute(default=datetime.utcnow)
    
    @classmethod
    def store_chat(
        cls,
        chat_id: str,
        match_id: str,
        initiator_id: str,
        responder_id: str,
        conversation_data: List[Dict[str, str]],
        ai_remarks: str = None,
        compatibility_score: int = None
    ) -> 'ChatRecord':
        """
        Store AI chat conversation in DynamoDB.
        
        Args:
            chat_id: Unique chat identifier
            match_id: Match identifier
            initiator_id: Initiator user ID
            responder_id: Responder user ID
            conversation_data: List of messages
            ai_remarks: AI-generated remarks
            compatibility_score: Compatibility score
            
        Returns:
            ChatRecord instance
        """
        # Convert conversation data to ChatMessage objects
        messages = []
        for msg in conversation_data:
            messages.append(
                ChatMessage(
                    sender_id=msg['sender_id'],
                    content=msg['content']
                )
            )
        
        chat_record = cls(
            chat_id=chat_id,
            match_id=match_id,
            initiator_id=initiator_id,
            responder_id=responder_id,
            conversation_data=messages,
            ai_remarks=ai_remarks,
            compatibility_score=compatibility_score
        )
        chat_record.save()
        
        return chat_record
