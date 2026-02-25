import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters.postgresql import postgresql_adapter
from app.adapters.dynamodb import UserProfile
from app.services.embedding_service import embedding_service

def regenerate_all_embeddings():
    users_processed = 0
    errors = 0
    try:
        # Fetch all user_ids that have persona
        logger.info("Fetching user profiles from DynamoDB...")
        # PynamoDB scan (simple, may be paginated in large tables)
        for profile in UserProfile.scan():
            user_id = profile.user_id
            persona = profile.persona
            if not persona or (not persona.requirements and not persona.offerings):
                continue
            requirements = persona.requirements or ""
            offerings = persona.offerings or ""
            try:
                logger.info(f"Regenerating embeddings for {user_id}")
                embedding_service.store_user_embeddings(user_id, requirements, offerings)
                users_processed += 1
            except Exception as e:
                logger.error(f"Failed regenerating embeddings for {user_id}: {e}")
                errors += 1
        logger.info(f"Completed regeneration. Users processed: {users_processed}, errors: {errors}")
    except Exception as e:
        logger.error(f"Script error: {e}")

if __name__ == '__main__':
    regenerate_all_embeddings()

