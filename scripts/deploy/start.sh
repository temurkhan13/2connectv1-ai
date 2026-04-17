#!/bin/bash
# Use `restart` not `start` — `systemctl start` on an already-running service
# is a no-op, and the AI service's `Restart=always` policy means systemd
# auto-restarts after stop.sh kills it, so `start` never actually cycles the
# process and new deployed code is not picked up. See Apr-18 Follow-up 25.
systemctl restart 2connect-ai
