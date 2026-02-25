"""
Test matching algorithm with 5+ diverse user profiles.
"""
import requests
import time
import uuid
from typing import Dict, List

AI_SERVICE = 'http://localhost:8000/api/v1'
API_KEY = 'dev-api-key'

# 5 diverse user profiles with different objectives and backgrounds
DIVERSE_USERS = [
    {
        "objective": "fundraising",
        "name": "Sarah (Fundraising)",
        "messages": [
            "I'm Sarah Chen, CEO of HealthAI. We're building an AI diagnostic platform for hospitals.",
            "We're pre-seed, targeting Series A in 6 months. Looking to raise $3M.",
            "Focus is B2B SaaS for healthcare. I have 10 years experience in medical tech."
        ]
    },
    {
        "objective": "investing",
        "name": "Michael (Investor)",
        "messages": [
            "I'm Michael Rivera, Partner at TechVentures Capital. We invest in early-stage startups.",
            "We write checks between $500K to $2M for seed and Series A rounds.",
            "Our focus is healthtech, fintech, and enterprise SaaS. Looking for strong technical founders."
        ]
    },
    {
        "objective": "hiring",
        "name": "Priya (Hiring)",
        "messages": [
            "I'm Priya Patel, VP Engineering at ScaleUp Inc. We're a 200-person startup.",
            "We need senior engineers, especially backend and ML specialists.",
            "Competitive salaries, remote-first culture. Growing 3x this year."
        ]
    },
    {
        "objective": "partnership",
        "name": "David (Partnership)",
        "messages": [
            "I'm David Kim, BD Director at CloudSync. We provide cloud infrastructure.",
            "Looking for startups who need reliable cloud services for their products.",
            "We offer startup credits and co-marketing opportunities."
        ]
    },
    {
        "objective": "mentorship",
        "name": "Jennifer (Mentor)",
        "messages": [
            "I'm Jennifer Wong, 20-year tech veteran, ex-CTO of BigTech.",
            "Happy to mentor first-time founders in B2B software.",
            "Focus areas: product-market fit, scaling engineering teams, fundraising strategy."
        ]
    },
]

def complete_onboarding(user_id: str, objective: str, messages: List[str]) -> dict:
    """Complete onboarding flow for a user via API."""
    headers = {'Content-Type': 'application/json', 'X-API-KEY': API_KEY}
    
    # Start session
    start_resp = requests.post(
        f'{AI_SERVICE}/onboarding/start',
        json={'user_id': user_id, 'objective': objective},
        headers=headers
    )
    if not start_resp.ok:
        return {'success': False, 'error': f'start failed: {start_resp.status_code}'}
    
    session_id = start_resp.json().get('session_id')
    
    # Chat messages
    for msg in messages:
        requests.post(
            f'{AI_SERVICE}/onboarding/chat',
            json={'user_id': user_id, 'session_id': session_id, 'message': msg},
            headers=headers
        )
        time.sleep(0.3)
    
    # Finalize
    requests.post(
        f'{AI_SERVICE}/onboarding/finalize/{session_id}',
        headers=headers
    )
    
    # Complete (triggers persona generation)
    complete_resp = requests.post(
        f'{AI_SERVICE}/onboarding/complete',
        json={'session_id': session_id, 'user_id': user_id},
        headers=headers
    )
    
    if complete_resp.ok:
        return {'success': True, 'session_id': session_id}
    return {'success': False, 'error': f'complete failed: {complete_resp.status_code}'}

def approve_summary(user_id: str) -> dict:
    """Approve summary and trigger embedding generation."""
    headers = {'Content-Type': 'application/json', 'X-API-KEY': API_KEY}
    
    resp = requests.post(
        f'{AI_SERVICE}/user/approve-summary',
        json={'user_id': user_id},
        headers=headers
    )
    
    if resp.ok:
        return {'success': True, 'data': resp.json()}
    return {'success': False, 'error': f'{resp.status_code}: {resp.text[:100]}'}

def check_matches(user_id: str) -> dict:
    """Check matches for a user."""
    headers = {'X-API-KEY': API_KEY}
    resp = requests.get(
        f'{AI_SERVICE}/matching/{user_id}/matches',
        headers=headers
    )
    if resp.ok:
        data = resp.json()
        matches = data.get('matches', data.get('data', {}).get('matches', []))
        return {'success': True, 'count': len(matches), 'matches': matches[:5]}
    return {'success': False, 'error': resp.status_code}

def main():
    print("=" * 70)
    print("DIVERSE MATCHING TEST - 5 Users with Different Objectives")
    print("=" * 70)
    
    user_ids = []
    
    # Create users via onboarding
    print("\n[PHASE 1] Creating diverse user profiles...")
    for i, user_data in enumerate(DIVERSE_USERS, 1):
        user_id = str(uuid.uuid4())
        print(f"\n  User {i}: {user_data['name']}")
        print(f"    ID: {user_id}")
        
        result = complete_onboarding(user_id, user_data['objective'], user_data['messages'])
        if result['success']:
            print(f"    [OK] Onboarding completed")
            user_ids.append({'id': user_id, 'name': user_data['name']})
        else:
            print(f"    [FAIL] {result.get('error')}")
    
    print(f"\n  Created {len(user_ids)} users successfully")
    
    # Wait for persona generation
    print("\n[PHASE 2] Waiting 25s for persona generation...")
    time.sleep(25)
    
    # Approve summaries (triggers embedding generation)
    print("\n[PHASE 3] Approving summaries (triggers embeddings & matching)...")
    for user in user_ids:
        result = approve_summary(user['id'])
        if result['success']:
            print(f"  [OK] {user['name']} - summary approved")
        else:
            print(f"  [FAIL] {user['name']} - {result.get('error')}")
    
    # Wait for embeddings and matching
    print("\n[PHASE 4] Waiting 20s for embeddings and matching...")
    time.sleep(20)
    
    # Check matches for each user
    print("\n[PHASE 5] Checking matches for each user...")
    total_matches = 0
    for user in user_ids:
        print(f"\n  {user['name']} ({user['id'][:8]}...)")
        result = check_matches(user['id'])
        if result['success']:
            print(f"    Matches found: {result['count']}")
            total_matches += result['count']
            for match in result['matches']:
                score = match.get('total_score', match.get('similarity_score', 'N/A'))
                target = match.get('user_id', match.get('target_user_id', 'N/A'))
                print(f"      - {target[:8]}... (score: {score:.3f})" if isinstance(score, float) else f"      - {target[:8]}... (score: {score})")
        else:
            print(f"    [ERROR] {result.get('error')}")
    
    print("\n" + "=" * 70)
    print(f"TOTAL MATCHES ACROSS ALL USERS: {total_matches}")
    print("=" * 70)
    
    # Return user IDs for cleanup
    return user_ids

if __name__ == '__main__':
    main()
