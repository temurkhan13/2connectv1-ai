#!/usr/bin/env python3
import os
import sys
import subprocess
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    project_dir = os.getcwd()
    print(f"Current directory: {project_dir}")
    
    # Get configuration from environment variables
    host = os.getenv('HOST')
    port = os.getenv('PORT')
    app_name = os.getenv('APP_NAME')
    
    if not all([host, port]):
        print("ERROR: Missing required environment variables: HOST, PORT")
        sys.exit(1)
    
    if os.path.exists("app"):
        print("Found 'app' directory - starting app.main")
        cmd = ["uv", "run", "uvicorn", "app.main:app", "--reload", "--host", host, "--port", port]
    else:
        print("'app' directory not found!")
        return
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Server will be available at: http://localhost:{port}")
    print(f"API docs will be available at: http://localhost:{port}/docs")
    print("Press Ctrl+C to stop the server")
    print("Make sure LocalStack is running: docker compose up -d")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()



