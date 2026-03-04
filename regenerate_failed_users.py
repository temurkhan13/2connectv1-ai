#!/usr/bin/env python3
"""Regenerate 6 Failed Users - Onboarding Fix Verification"""

import requests
import time
import json

BACKEND_URL = "https://twoconnectv1-backend.onrender.com/api/v1"
AI_SERVICE_URL = "https://twoconnectv1-ai.onrender.com/api/v1"

FAILED_USER_EMAILS = [
    "jaredwbonham@gmail.com",
    "jshukran@gmail.com",
    "jose@2connect.ai",
    "shane@2connect.ai",
    "ryan@stbl.io",
    "rybest@gmail.com"
]

def get_user_id(email):
    try:
        r = requests.get(f"{BACKEND_URL}/users/by-email/{email}", timeout=10)
        if r.status_code == 200:
            return r.json().get("data", {}).get("id")
        print(f"[ERROR] Could not fetch user ID: {r.status_code}")
    except Exception as e:
        print(f"[ERROR] {e}")
    return None

def trigger_regen(user_id):
    try:
        r = requests.post(
            f"{AI_SERVICE_URL}/personas/regenerate",
            json={"user_id": user_id},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if r.status_code in [200, 201, 202]:
            print(f"[OK] Triggered for {user_id[:8]}...")
            return True
        print(f"[ERROR] Failed: {r.status_code}")
    except Exception as e:
        print(f"[ERROR] {e}")
    return False

def check_status(user_id):
    try:
        r = requests.get(f"{AI_SERVICE_URL}/personas/{user_id}", timeout=10)
        if r.status_code == 200:
            return r.json().get("data")
    except:
        pass
    return None

def regenerate_user(email, idx, total):
    print(f"\n{'='*60}")
    print(f"User {idx}/{total}: {email}")
    print('='*60)
    
    result = {"email": email, "user_id": None, "success": False}
    
    print(f"[INFO] Fetching user ID...")
    user_id = get_user_id(email)
    if not user_id:
        return result
    
    result["user_id"] = user_id
    print(f"[OK] User ID: {user_id}")
    
    print(f"[INFO] Triggering regeneration...")
    if not trigger_regen(user_id):
        return result
    
    result["success"] = True
    print(f"[INFO] Waiting 5s...")
    time.sleep(5)
    
    print(f"[INFO] Checking status...")
    status = check_status(user_id)
    if status:
        result["status"] = status.get("status")
        print(f"[OK] Status: {result['status']}")
    
    return result

def main():
    print("\n" + "="*60)
    print("REGENERATE 6 FAILED USERS".center(60))
    print("="*60)
    print(f"Backend: {BACKEND_URL}")
    print(f"AI Service: {AI_SERVICE_URL}\n")
    
    results = []
    for i, email in enumerate(FAILED_USER_EMAILS, 1):
        result = regenerate_user(email, i, len(FAILED_USER_EMAILS))
        results.append(result)
        if i < len(FAILED_USER_EMAILS):
            time.sleep(3)
    
    print("\n" + "="*60)
    print("SUMMARY".center(60))
    print("="*60)
    
    success = sum(1 for r in results if r["success"])
    print(f"\nTotal: {len(FAILED_USER_EMAILS)}")
    print(f"Success: {success}")
    print(f"Failed: {len(FAILED_USER_EMAILS)-success}\n")
    
    for r in results:
        status = "[OK]" if r["success"] else "[FAIL]"
        print(f"{status} {r['email']}")
        print(f"      User ID: {r['user_id'] or 'N/A'}")
        print(f"      Status: {r.get('status', 'Unknown')}\n")
    
    with open("regeneration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("[OK] Results saved to: regeneration_results.json")
    print("\nComplete!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
