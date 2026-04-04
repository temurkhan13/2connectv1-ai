import os
import sys
import json
import time
from dotenv import load_dotenv
load_dotenv('.env.production', override=True)

from app.services.inline_matching_service import inline_matching_service

PROGRESS_FILE = "scripts/.local_regen_progress.json"

def load_done():
    if os.path.exists(PROGRESS_FILE):
        return set(json.load(open(PROGRESS_FILE)).get("done", []))
    return set()

def save_done(done):
    json.dump({"done": list(done)}, open(PROGRESS_FILE, "w"), indent=2)

ids = [l.strip() for l in open("scripts/user_ids.txt") if l.strip()]
done = load_done()
remaining = [uid for uid in ids if uid not in done]

print(f"Total: {len(ids)}, Done: {len(done)}, Remaining: {len(remaining)}")

for i, uid in enumerate(remaining):
    print(f"[{len(done)+i+1}/{len(ids)}] {uid[:12]}...", end=" ", flush=True)
    try:
        result = inline_matching_service.calculate_and_sync_matches_bidirectional(uid, threshold=0.5)
        matches = result.get("new_user_matches", result.get("matches_count", "?"))
        print(f"OK — {matches} matches")
        done.add(uid)
        save_done(done)
    except KeyboardInterrupt:
        print("\nStopped. Progress saved.")
        save_done(done)
        sys.exit(0)
    except Exception as e:
        print(f"FAILED: {e}")
    time.sleep(2)

print(f"\nDone: {len(done)}/{len(ids)}")
