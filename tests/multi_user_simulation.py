import os
import sys
import logging
from typing import Dict, Any, List
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ai_chat_service import AIChatService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class TestAIChatService(AIChatService):
    """
    Subclass to override get_persona and check_alignment for testing without DB.
    """
    def __init__(self):
        super().__init__()
        # Mock DB or just don't use it since we override get_persona
        self.personas_db = {}
        self.alignments_db = {}

    def get_persona(self, user_id: str) -> Dict[str, Any]:
        return self.personas_db.get(user_id, {})

    def check_alignment(self, persona_a: Dict[str, Any], persona_b: Dict[str, Any]) -> Dict[str, Any]:
        # Return pre-defined alignment based on IDs or just calculate mock
        # For this test, we'll map IDs to alignment results
        key = f"{persona_a['id']}-{persona_b['id']}"
        return self.alignments_db.get(key, {"alignment_type": "unknown", "alignment_score": 50})
    
    def calculate_compatibility_score(self, *args, **kwargs):
        return {"score": 80, "analysis": "Mock analysis"}
    
    def generate_conversation_summary(self, *args, **kwargs):
        return "Mock summary"

def run_simulation_test():
    service = TestAIChatService()
    if not service.client:
        print("SKIPPING: No OpenAI Client")
        return

    # --- SCENARIO 1: FULLY ALIGNED ---
    founder = {
        "id": "founder1", "name": "Alice", "designation": "Founder",
        "requirements": "Seeking $500k Seed funding for AI startup.",
        "offerings": "Equity, High Growth"
    }
    investor = {
        "id": "investor1", "name": "Bob", "designation": "Investor",
        "requirements": "Looking for AI startups to invest in.",
        "offerings": "Capital, Mentorship"
    }
    
    service.personas_db["founder1"] = founder
    service.personas_db["investor1"] = investor
    service.alignments_db["founder1-investor1"] = {"alignment_type": "fully_aligned", "alignment_score": 85}
    
    print("\n" + "="*50)
    print("SCENARIO 1: FULLY ALIGNED (Founder + Investor)")
    print("="*50)
    result = service.simulate_conversation("founder1", "investor1", "match1")
    print_conversation(result["conversation_data"])

    # --- SCENARIO 2: MISALIGNED (Loop Check) ---
    chef = {
        "id": "chef1", "name": "Charlie", "designation": "Chef",
        "requirements": "Looking for restaurant partners.",
        "offerings": "Culinary skills"
    }
    tech_investor = {
        "id": "investor2", "name": "Dave", "designation": "Tech VC",
        "requirements": "Investing in SaaS only.",
        "offerings": "Tech Capital"
    }
    
    service.personas_db["chef1"] = chef
    service.personas_db["investor2"] = tech_investor
    service.alignments_db["chef1-investor2"] = {"alignment_type": "misaligned", "alignment_score": 15}
    
    print("\n" + "="*50)
    print("SCENARIO 2: MISALIGNED (Chef + Tech VC)")
    print("="*50)
    result = service.simulate_conversation("chef1", "investor2", "match2")
    print_conversation(result["conversation_data"])

def print_conversation(conversation_data):
    for msg in conversation_data:
        sender = msg['sender_id']
        # Map ID to name for readability
        name = "Unknown"
        if "founder" in sender or "Alice" in sender: name = "Alice (Founder)"
        elif "investor1" in sender or "Bob" in sender: name = "Bob (Investor)"
        elif "chef" in sender or "Charlie" in sender: name = "Charlie (Chef)"
        elif "investor2" in sender or "Dave" in sender: name = "Dave (Tech VC)"
        
        print(f"[{name}]: {msg['content']}")

if __name__ == "__main__":
    run_simulation_test()
