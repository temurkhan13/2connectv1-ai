#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    project_dir = os.getcwd()
    print(f"Current directory: {project_dir}")

    if os.path.exists("app"):
        print("Found 'app' directory - starting Celery worker + beat scheduler")
        cmd = ["uv", "run", "celery", "-A", "app.core.celery", "worker", "--beat", "--loglevel=info"]
    else:
        print("'app' directory not found!")
        return

    print(f"Running command: {' '.join(cmd)}")
    print("Celery worker will process background tasks + scheduled tasks")
    print("Make sure Redis is running: docker compose up -d")
    print("")
    print("Celery worker logs:")
    print("- Processing tasks for resume parsing, persona generation, and matching")
    print("- Running scheduled matchmaking task (check celery.py for schedule)")
    print("- Check this terminal for task execution logs")
    print("")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nCelery worker stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting Celery worker: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
