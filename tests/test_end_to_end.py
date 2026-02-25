#!/usr/bin/env python3
"""
End-to-end test for the AI-powered matchmaking system.

This test validates the complete workflow:
1. User registration without resume (optional resume feature)
2. Celery task chain execution
3. Persona generation from questions
4. Embedding generation and storage
5. Matchmaking functionality
6. API endpoint validation

Usage:
    python tests/test_end_to_end.py
"""

import os
import sys
import time
import json
import requests
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EndToEndTester:
    """End-to-end system tester."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the tester."""
        self.base_url = base_url
        self.registered_users = []
        self.session = requests.Session()
        
    def log_step(self, step: str):
        """Log a test step."""
        logger.info(f"STEP: {step}")
    
    def log_result(self, success: bool, message: str):
        """Log test result."""
        if success:
            logger.info(f"SUCCESS: {message}")
        else:
            logger.error(f"FAILED: {message}")
    
    def log_info(self, message: str):
        """Log information."""
        logger.info(f"INFO: {message}")
    
    def check_api_health(self) -> bool:
        """Verify API is running and healthy."""
        self.log_step("Checking API health")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            success = response.status_code == 200
            self.log_result(success, f"API health check - Status: {response.status_code}")
            return success
        except Exception as e:
            self.log_result(False, f"API health check failed: {str(e)}")
            return False
    
    def register_user(self, user_id: str, questions: List[Dict[str, str]]) -> bool:
        """Register a user without resume."""
        self.log_step(f"Registering user: {user_id}")
        
        payload = {
            "user_id": user_id,
            "questions": questions
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/user/register",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                message = result.get("message", "No message")
                
                if success:
                    self.registered_users.append(user_id)
                    self.log_result(True, f"User {user_id} registered: {message}")
                    return True
                else:
                    self.log_result(False, f"Registration failed: {message}")
                    return False
            else:
                self.log_result(False, f"Registration failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result(False, f"Registration failed: {str(e)}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/user/{user_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def wait_for_processing(self, user_id: str, timeout: int = 180) -> bool:
        """Wait for user processing to complete."""
        self.log_step(f"Waiting for processing completion: {user_id}")
        
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout:
            profile = self.get_user_profile(user_id)
            
            if not profile:
                time.sleep(5)
                continue
            
            processing_status = profile.get("processing_status", "unknown")
            persona_status = profile.get("persona_status", "unknown")
            current_status = f"processing={processing_status}, persona={persona_status}"
            
            if current_status != last_status:
                self.log_info(f"Status update for {user_id}: {current_status}")
                last_status = current_status
            
            # Check completion
            if (processing_status in ["completed", "failed"] and 
                persona_status in ["completed", "failed"]):
                
                if processing_status == "completed" and persona_status == "completed":
                    persona = profile.get("persona", {})
                    persona_name = persona.get("name", "Unknown")
                    self.log_result(True, f"Processing completed for {user_id}: {persona_name}")
                    return True
                else:
                    self.log_result(False, f"Processing failed for {user_id}: {current_status}")
                    return False
            
            time.sleep(5)
        
        self.log_result(False, f"Processing timeout after {timeout}s for {user_id}")
        return False
    
    def test_matching(self, user_id: str) -> bool:
        """Test matching functionality."""
        self.log_step(f"Testing matching for user: {user_id}")
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/matching/{user_id}/matches",
                params={"top_k": 10, "similarity_threshold": 0.2},
                timeout=30
            )
            
            if response.status_code == 200:
                matches = response.json()
                total_matches = matches.get("total_matches", 0)
                requirements_matches = len(matches.get("requirements_matches", []))
                offerings_matches = len(matches.get("offerings_matches", []))
                
                self.log_info(f"Matches for {user_id}: total={total_matches}, requirements={requirements_matches}, offerings={offerings_matches}")
                
                # Log sample matches
                if requirements_matches > 0:
                    sample = matches["requirements_matches"][0]
                    self.log_info(f"Sample requirements match: {sample.get('user_id')} (score: {sample.get('similarity_score', 0):.3f})")
                
                if offerings_matches > 0:
                    sample = matches["offerings_matches"][0]
                    self.log_info(f"Sample offerings match: {sample.get('user_id')} (score: {sample.get('similarity_score', 0):.3f})")
                
                self.log_result(True, f"Matching test completed for {user_id}")
                return True
            else:
                self.log_result(False, f"Matching failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result(False, f"Matching test failed: {str(e)}")
            return False
    
    def test_matching_stats(self) -> bool:
        """Test matching statistics endpoint."""
        self.log_step("Testing matching statistics")
        
        try:
            response = self.session.get(f"{self.base_url}/api/v1/matching/stats", timeout=10)
            
            if response.status_code == 200:
                stats = response.json()
                self.log_info(f"Matching statistics: {json.dumps(stats, indent=2)}")
                self.log_result(True, "Matching statistics retrieved")
                return True
            else:
                self.log_result(False, f"Statistics failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result(False, f"Statistics test failed: {str(e)}")
            return False
    
    def create_test_data(self) -> List[Dict[str, Any]]:
        """Create test user data."""
        return [
            {
                "user_id": "developer_001",
                "questions": [
                    {
                        "prompt": "What are you looking for?",
                        "answer": "I need seed funding and mentorship for my AI startup. Looking for investors who understand machine learning and can provide strategic guidance for scaling technology products."
                    },
                    {
                        "prompt": "What can you offer?",
                        "answer": "Full-stack development expertise with 5+ years experience. Specialized in Python, React, and machine learning systems. Can build scalable web applications and AI-powered products from concept to production."
                    },
                    {
                        "prompt": "What is your experience?",
                        "answer": "Senior software engineer with experience at tech startups. Built multiple AI products including recommendation systems and natural language processing applications. Strong background in system architecture and team leadership."
                    },
                    {
                        "prompt": "What industry focus?",
                        "answer": "Technology sector focused on artificial intelligence and business automation. Interested in B2B SaaS products that solve real business problems using machine learning."
                    }
                ]
            },
            {
                "user_id": "investor_001",
                "questions": [
                    {
                        "prompt": "What are you looking for?",
                        "answer": "Early-stage technology startups with strong technical teams and proven MVP. Particularly interested in AI/ML companies with clear market validation and scalable business models."
                    },
                    {
                        "prompt": "What can you offer?",
                        "answer": "Seed funding from $100K to $500K, extensive network of industry contacts, and hands-on mentorship. 10+ years of startup experience including two successful exits in the technology sector."
                    },
                    {
                        "prompt": "What is your investment focus?",
                        "answer": "Technology companies, especially AI/ML, fintech, and B2B SaaS. I prefer technical founders with domain expertise and look for companies that can scale efficiently with technology leverage."
                    },
                    {
                        "prompt": "What is your background?",
                        "answer": "Former startup founder with successful exits. Now angel investor and advisor to 20+ companies. Strong technical background with experience in product development and go-to-market strategy."
                    }
                ]
            },
            {
                "user_id": "designer_001",
                "questions": [
                    {
                        "prompt": "What are you looking for?",
                        "answer": "Technical co-founder who can build beautiful, user-friendly products. Need someone with strong engineering skills and appreciation for design quality to create consumer-facing applications."
                    },
                    {
                        "prompt": "What can you offer?",
                        "answer": "10+ years of UX/UI design experience with focus on mobile and web applications. Expert in user research, product strategy, and design systems. Can lead entire product design process from concept to launch."
                    },
                    {
                        "prompt": "What type of products?",
                        "answer": "Consumer-facing applications with emphasis on user experience and visual design. Interested in products that solve everyday problems through intuitive interfaces and delightful interactions."
                    },
                    {
                        "prompt": "What is your goal?",
                        "answer": "Build a design-first startup that prioritizes user experience. Looking for technical partnership to create innovative products that users love and find indispensable in their daily lives."
                    }
                ]
            }
        ]
    
    def run_end_to_end_test(self) -> bool:
        """Execute complete end-to-end test."""
        logger.info("Starting end-to-end system test")
        logger.info("=" * 80)
        
        # Test 1: API Health Check
        if not self.check_api_health():
            return False
        
        # Test 2: User Registration
        test_users = self.create_test_data()
        logger.info(f"Testing with {len(test_users)} users (no resume - optional resume feature)")
        
        registration_success = True
        for user_data in test_users:
            if not self.register_user(user_data["user_id"], user_data["questions"]):
                registration_success = False
        
        if not registration_success:
            self.log_result(False, "User registration phase failed")
            return False
        
        # Test 3: Processing Completion
        logger.info("Waiting for initial processing to start...")
        time.sleep(10)
        
        processing_success = True
        for user_data in test_users:
            if not self.wait_for_processing(user_data["user_id"]):
                processing_success = False
        
        if not processing_success:
            self.log_result(False, "Processing completion phase failed")
            return False
        
        # Test 4: Embedding Generation Wait
        logger.info("Waiting for embedding generation to complete...")
        time.sleep(20)
        
        # Test 5: Matching Tests
        matching_success = True
        for user_data in test_users:
            if not self.test_matching(user_data["user_id"]):
                matching_success = False
        
        if not matching_success:
            self.log_result(False, "Matching phase failed")
            return False
        
        # Test 6: Statistics Test
        if not self.test_matching_stats():
            return False
        
        # Final Results
        logger.info("=" * 80)
        logger.info("END-TO-END TEST COMPLETED SUCCESSFULLY")
        logger.info(f"Tested {len(test_users)} users through complete workflow:")
        logger.info("- User registration without resume")
        logger.info("- Celery task chain execution")
        logger.info("- Persona generation from questions")
        logger.info("- Embedding generation and storage")
        logger.info("- Matchmaking functionality")
        logger.info("- API endpoint validation")
        logger.info("=" * 80)
        
        return True

def main():
    """Main test execution."""
    tester = EndToEndTester()
    
    try:
        success = tester.run_end_to_end_test()
        if success:
            logger.info("Test suite completed successfully")
            sys.exit(0)
        else:
            logger.error("Test suite failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
