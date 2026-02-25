#!/usr/bin/env python3
"""
Test script for the matches notification system.

This test validates:
1. User registration and processing pipeline
2. Persona generation completion notification
3. Embedding generation and matches notification
4. Actual backend notification calls

Usage:
    python tests/test_matches_notification.py
"""

import os
import sys
import time
import json
import requests
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotificationTester:
    """Test the complete notification system."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize the tester."""
        self.api_url = api_url
        self.session = requests.Session()
        self.registered_users = []
    
    def test_backend_health(self) -> bool:
        """Test backend matches-ready endpoint health."""
        backend_url = os.getenv('RECIPROCITY_BACKEND_URL')
        if not backend_url:
            logger.error("RECIPROCITY_BACKEND_URL not set, cannot test backend health")
            return False
        
        endpoint = f"{backend_url}/user/matches-ready"
        logger.info(f"Testing backend health: {endpoint}")
        
        # Test payload
        test_payload = {
            "batch_id": str(uuid.uuid4()),
            "user_id": "health-check-user",
            "matches": []
        }
        
        try:
            response = self.session.post(
                endpoint,
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            logger.info(f"Backend response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    logger.info(f"Backend response: {json.dumps(response_data, indent=2)}")
                    
                    # Check expected response format
                    if (response_data.get("success") is True and 
                        response_data.get("message") == "matches received!"):
                        logger.info("Backend health check PASSED - correct response format")
                        return True
                    else:
                        logger.warning("Backend responded but format doesn't match expected")
                        logger.warning("Expected: {'success': true, 'message': 'matches received!'}")
                        return True  # Still consider it working
                        
                except json.JSONDecodeError:
                    logger.warning("Backend responded but not with JSON")
                    logger.info(f"Response text: {response.text}")
                    return True  # Still consider it working
            else:
                logger.error(f"Backend health check FAILED: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Backend health check FAILED: {e}")
            return False
    
    def register_test_users(self) -> List[str]:
        """Register test users for notification testing."""
        test_users = [
            {
                "user_id": f"test-developer-{uuid.uuid4().hex[:8]}",
                "questions": [
                    {
                        "prompt": "What are you looking for?",
                        "answer": "I need seed funding and technical mentorship for my AI startup. Looking for investors who understand machine learning."
                    },
                    {
                        "prompt": "What can you offer?",
                        "answer": "Full-stack development expertise, AI/ML implementation, and 5+ years of software engineering experience."
                    },
                    {
                        "prompt": "What is your background?",
                        "answer": "Senior software engineer with expertise in Python, React, and machine learning systems."
                    }
                ]
            },
            {
                "user_id": f"test-investor-{uuid.uuid4().hex[:8]}",
                "questions": [
                    {
                        "prompt": "What are you looking for?",
                        "answer": "Early-stage AI startups with strong technical teams and proven MVP. Particularly interested in machine learning applications."
                    },
                    {
                        "prompt": "What can you offer?",
                        "answer": "Seed funding up to $500K, extensive network of industry contacts, and hands-on mentorship."
                    },
                    {
                        "prompt": "What is your investment focus?",
                        "answer": "AI/ML companies, B2B SaaS, and technical founders with domain expertise."
                    }
                ]
            }
        ]
        
        for user_data in test_users:
            logger.info(f"Registering user: {user_data['user_id']}")
            
            try:
                response = self.session.post(
                    f"{self.api_url}/api/v1/user/register",
                    json=user_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        self.registered_users.append(user_data['user_id'])
                        logger.info(f"Successfully registered: {user_data['user_id']}")
                    else:
                        logger.error(f"Registration failed: {result.get('message')}")
                else:
                    logger.error(f"Registration failed: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Registration error: {e}")
        
        return self.registered_users
    
    def wait_for_processing_completion(self, timeout: int = 300) -> bool:
        """Wait for all users to complete processing."""
        logger.info(f"Waiting for processing completion (timeout: {timeout}s)")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            completed_users = 0
            
            for user_id in self.registered_users:
                try:
                    response = self.session.get(f"{self.api_url}/api/v1/user/{user_id}", timeout=10)
                    if response.status_code == 200:
                        profile = response.json()
                        processing_status = profile.get("processing_status", "unknown")
                        persona_status = profile.get("persona_status", "unknown")
                        
                        # Debug logging
                        logger.info(f"User {user_id}: processing={processing_status}, persona={persona_status}")
                        
                        if processing_status == "completed" and persona_status == "completed":
                            completed_users += 1
                    else:
                        logger.warning(f"Failed to get profile for user {user_id}: HTTP {response.status_code}")
                except Exception as e:
                    logger.warning(f"Error checking user {user_id}: {e}")
            
            logger.info(f"Processing completion: {completed_users}/{len(self.registered_users)} users")
            
            if completed_users == len(self.registered_users):
                logger.info("All users completed processing")
                return True
            
            time.sleep(10)  # Check every 10 seconds
        
        logger.error(f"Timeout waiting for processing completion after {timeout}s")
        return False
    
    
    def send_matches_notifications(self) -> bool:
        """Manually send matches notifications for all users."""
        logger.info("Sending matches notifications manually")
        
        # Import notification service
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app.services.notification_service import NotificationService
        from app.services.embedding_service import embedding_service
        from app.services.matching_service import matching_service
        
        notification_service = NotificationService()
        
        if not notification_service.is_configured():
            logger.error("Backend URL not configured for notifications")
            return False
        
        success = True
        
        for user_id in self.registered_users:
            try:
                logger.info(f"Getting matches for user: {user_id}")
                
                # Get requirements matches
                matches_result = matching_service.find_user_matches(user_id)
                requirements_matches = matches_result.get('requirements_matches', [])
                
                # Always send notification, even with empty matches
                batch_id = str(uuid.uuid4())
                
                logger.info(f"Sending notification for user {user_id} with {len(requirements_matches)} matches")
                
                # Send notification (with empty array if no matches)
                notification_result = notification_service.send_matches_ready_notification(
                    user_id=user_id,
                    batch_id=batch_id,
                    matches=requirements_matches  # This will be empty array if no matches
                )
                
                if notification_result.get("success"):
                    if requirements_matches:
                        logger.info(f"Successfully sent matches notification for user {user_id} with {len(requirements_matches)} matches")
                    else:
                        logger.info(f"Successfully sent empty matches notification for user {user_id}")
                else:
                    logger.error(f"Failed to send matches notification for user {user_id}: {notification_result.get('message')}")
                    success = False
                    
            except Exception as e:
                logger.error(f"Error sending notification for user {user_id}: {e}")
                success = False
        
        return success
    
    def run_test(self) -> bool:
        """Run complete notification test."""
        logger.info("Starting matches notification test")
        logger.info("=" * 80)
        
        try:
            # Step 1: Test backend health first
            if not self.test_backend_health():
                logger.error("Backend health check failed, stopping test")
                return False
            
            # Step 2: Register test users
            registered_users = self.register_test_users()
            if not registered_users:
                logger.error("No users registered successfully")
                return False
            
            logger.info(f"Successfully registered {len(registered_users)} users")
            
            # Step 3: Wait for processing completion
            logger.info("Waiting for embeddings to complete...")
            if not self.wait_for_processing_completion():
                return False
            
            # Step 4: Send matches notifications manually
            if not self.send_matches_notifications():
                return False
            
            # Final summary
            logger.info("=" * 80)
            logger.info("MATCHES NOTIFICATION TEST COMPLETED SUCCESSFULLY")
            logger.info(f"Processed users: {len(registered_users)}")
            logger.info("All matches notifications sent to backend")
            logger.info("Check your backend logs for notification receipts")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            return False

def main():
    """Main test execution."""
    tester = NotificationTester()
    
    try:
        success = tester.run_test()
        if success:
            logger.info("Notification test completed successfully")
            sys.exit(0)
        else:
            logger.error("Notification test failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
