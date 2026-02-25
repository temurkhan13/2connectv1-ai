import os
import sys
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ai_chat_service import AIChatService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_test_case(scenario_name: str, initiator: Dict[str, Any], responder: Dict[str, Any], alignment_info: Dict[str, Any]):
    print(f"\n{'='*20} {scenario_name} {'='*20}")
    
    service = AIChatService()
    
    if not service.client:
        print("SKIPPING: No OpenAI Client")
        return

    try:
        # 1. Initiator starts
        print(f"\n[INITIATOR] {initiator['name']} ({initiator['designation']}):")
        msg1 = service.generate_ai_response(
            conversation_history=[],
            persona=initiator,
            other_persona=responder,
            role="initiator",
            stage="introduction",
            alignment_info=alignment_info
        )
        print(f"> {msg1}")
        
        # 2. Responder replies
        print(f"\n[RESPONDER] {responder['name']} ({responder['designation']}):")
        history = [{"role": "user", "content": msg1}]
        msg2 = service.generate_ai_response(
            conversation_history=history,
            persona=responder,
            other_persona=initiator,
            role="responder",
            stage="introduction",
            alignment_info=alignment_info
        )
        print(f"> {msg2}")
        
        # 3. Initiator replies back
        print(f"\n[INITIATOR] {initiator['name']} (Second Turn):")
        history_initiator_turn2 = [
            {"role": "assistant", "content": msg1}, 
            {"role": "user", "content": msg2}       
        ]
        
        msg3 = service.generate_ai_response(
            conversation_history=history_initiator_turn2,
            persona=initiator,
            other_persona=responder,
            role="initiator",
            stage="discovery", 
            alignment_info=alignment_info
        )
        print(f"> {msg3}")
        
        # 4. Responder finalizes (Testing "No Fake Emails")
        print(f"\n[RESPONDER] {responder['name']} (Second Turn):")
        history_responder_turn2 = [
            {"role": "user", "content": msg1}, 
            {"role": "assistant", "content": msg2},
            {"role": "user", "content": msg3}
        ]
        
        msg4 = service.generate_ai_response(
            conversation_history=history_responder_turn2,
            persona=responder,
            other_persona=initiator,
            role="responder",
            stage="deep_dive",
            alignment_info=alignment_info
        )
        print(f"> {msg4}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    # Test 4: No Fake Emails
    edtech_seeker = {
        "name": "EdTech Seeker", "designation": "Director",
        "requirements": "Seeking partnerships with learning institutions and EdTech startups.",
        "offerings": "Digital learning solutions."
    }
    consultant = {
        "name": "Consultant", "designation": "Partner",
        "requirements": "Consulting projects.",
        "offerings": "Bundle partner solutions into consulting projects. Enterprise client base."
    }
    
    # Assuming aligned
    run_test_case("Test 4: No Fake Emails Check", edtech_seeker, consultant, {"alignment_type": "fully_aligned", "alignment_score": 80})

if __name__ == "__main__":
    main()
