"""
Safe Match Regeneration Script — Survives Render Restarts

Calls the per-user /admin/regenerate-matches endpoint one user at a time.
Tracks progress in a local file so it can resume from where it left off
if interrupted (Render restart, network error, etc).

Usage:
    # Dry run — list users, don't regenerate
    python scripts/regenerate_matches_safe.py --dry-run

    # Run for real
    python scripts/regenerate_matches_safe.py

    # Resume after interruption (automatically skips completed users)
    python scripts/regenerate_matches_safe.py

    # Force restart from scratch
    python scripts/regenerate_matches_safe.py --reset
"""
import requests
import json
import time
import sys
import os
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

AI_SERVICE_URL = "https://twoconnectv1-ai.onrender.com/api/v1"
ADMIN_KEY = "migrate-2connect-2026"
PROGRESS_FILE = "scripts/.match_regen_progress.json"
DELAY_BETWEEN_USERS = 5  # seconds — give Render breathing room

# ─── Progress Tracking ────────────────────────────────────────────────────────

def load_progress():
    """Load progress from file. Returns set of completed user IDs."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
            return set(data.get("completed", []))
    return set()


def save_progress(completed: set, total: int):
    """Save progress to file."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "completed": list(completed),
            "total": total,
            "last_updated": datetime.utcnow().isoformat(),
        }, f, indent=2)


def reset_progress():
    """Delete progress file to start fresh."""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress reset.")


# ─── Fetch All Users ──────────────────────────────────────────────────────────

def get_all_user_ids():
    """Fetch all completed user IDs from the AI service."""
    # Use the system-health endpoint to get user count, then scan profiles
    # The reembed endpoint in dry-run mode gives us the count but not IDs
    # Use the admin endpoint that lists users
    try:
        # Try the admin system-health to get user count
        resp = requests.get(
            f"{AI_SERVICE_URL}/admin/system-health",
            headers={"X-API-KEY": ADMIN_KEY},
            timeout=30,
        )
        if resp.status_code == 200:
            print(f"Service healthy. Fetching user list...")
    except Exception as e:
        print(f"Warning: Could not check health: {e}")

    # Get users from the reembed dry-run (it scans all profiles)
    # We need actual user IDs — let's use the admin endpoint
    try:
        resp = requests.get(
            f"{AI_SERVICE_URL}/admin/users",
            headers={"X-API-KEY": ADMIN_KEY},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            users = data.get("data", data.get("result", data.get("users", [])))
            if isinstance(users, list):
                user_ids = []
                for u in users:
                    uid = u.get("user_id") or u.get("id") or u.get("userId")
                    status = u.get("onboarding_status") or u.get("status", "")
                    # Only include users who completed onboarding
                    if uid and status in ("completed", "complete", "onboarded"):
                        user_ids.append(uid)
                if user_ids:
                    return user_ids
                # If no status filter, return all
                return [u.get("user_id") or u.get("id") or u.get("userId") for u in users if u.get("user_id") or u.get("id")]
    except Exception as e:
        print(f"Could not fetch from /admin/users: {e}")

    # Fallback: try /admin/all-users or similar
    for endpoint in ["/admin/all-users", "/admin/list-users", "/users"]:
        try:
            resp = requests.get(
                f"{AI_SERVICE_URL}{endpoint}",
                headers={"X-API-KEY": ADMIN_KEY},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                users = data.get("data", data.get("result", []))
                if isinstance(users, list) and len(users) > 0:
                    return [u.get("user_id") or u.get("id") for u in users if u.get("user_id") or u.get("id")]
        except:
            continue

    print("\nERROR: Could not fetch user list from any endpoint.")
    print("You can manually provide user IDs by creating a file 'scripts/user_ids.txt'")
    print("with one user ID per line.")

    # Try manual file
    manual_file = "scripts/user_ids.txt"
    if os.path.exists(manual_file):
        with open(manual_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
            if ids:
                print(f"Loaded {len(ids)} user IDs from {manual_file}")
                return ids

    return []


# ─── Regenerate Per User ──────────────────────────────────────────────────────

def regenerate_for_user(user_id: str) -> dict:
    """Call the per-user regenerate-matches endpoint."""
    resp = requests.post(
        f"{AI_SERVICE_URL}/admin/regenerate-matches",
        json={"admin_key": ADMIN_KEY, "user_id": user_id, "threshold": 0.5},
        timeout=300,  # 5 min timeout per user
    )

    if resp.status_code == 200:
        return resp.json()
    else:
        return {"error": resp.status_code, "detail": resp.text[:200]}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    dry_run = "--dry-run" in sys.argv
    reset = "--reset" in sys.argv

    if reset:
        reset_progress()

    print("=" * 60)
    print("SAFE MATCH REGENERATION — Survives Render Restarts")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Target: {AI_SERVICE_URL}")
    print()

    # Load progress
    completed = load_progress()
    if completed:
        print(f"Resuming — {len(completed)} users already completed")

    # Get all user IDs
    print("Fetching user list...")
    all_users = get_all_user_ids()

    if not all_users:
        print("No users found. Exiting.")
        return

    # Filter out already completed
    remaining = [uid for uid in all_users if uid not in completed]
    total = len(all_users)

    print(f"Total users: {total}")
    print(f"Already done: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print()

    if dry_run:
        print("DRY RUN — would regenerate matches for:")
        for i, uid in enumerate(remaining[:10]):
            print(f"  {i+1}. {uid}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
        return

    # Process one at a time
    failed = []
    for i, user_id in enumerate(remaining):
        idx = len(completed) + i + 1
        print(f"[{idx}/{total}] Regenerating matches for {user_id[:12]}...", end=" ", flush=True)

        try:
            result = regenerate_for_user(user_id)

            if "error" in result:
                print(f"ERROR {result['error']}: {result.get('detail', '')[:80]}")
                failed.append(user_id)

                # If 502/503, Render might be restarting — wait longer
                if result.get("error") in (502, 503):
                    print("  Render may be restarting. Waiting 30s...")
                    time.sleep(30)
            else:
                match_count = result.get("result", {}).get("matches_count",
                              result.get("result", {}).get("new_user_matches", "?"))
                print(f"OK — {match_count} matches")
                completed.add(user_id)
                save_progress(completed, total)

        except requests.exceptions.Timeout:
            print("TIMEOUT (5min) — skipping, will retry on next run")
            failed.append(user_id)
        except requests.exceptions.ConnectionError:
            print("CONNECTION ERROR — Render may be down. Waiting 60s...")
            failed.append(user_id)
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n\nInterrupted! Progress saved. Run again to resume.")
            save_progress(completed, total)
            return
        except Exception as e:
            print(f"UNEXPECTED ERROR: {e}")
            failed.append(user_id)

        # Delay between users
        if i < len(remaining) - 1:
            time.sleep(DELAY_BETWEEN_USERS)

    # Summary
    print()
    print("=" * 60)
    print("COMPLETE")
    print(f"  Processed: {len(completed)}/{total}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed IDs: {failed[:5]}{'...' if len(failed) > 5 else ''}")
        print("  Run the script again to retry failed users.")
    print("=" * 60)


if __name__ == "__main__":
    main()
