"""
Launch: python server.py
Opens at: http://localhost:8000/app
"""
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Load environment variables from .env file (searches parent dirs too)
try:
    from dotenv import load_dotenv
    # Try current dir, then parent dir (where .env lives at clone root)
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded .env from {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

# Import your existing app
from main import app


frontend_dir = Path(__file__).parent / "frontend"

# Mount static assets
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/app", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(frontend_dir / "index.html"))

if __name__ == "__main__":
    print("\n⚖️  Legal Citation Auditor")
    print("   Frontend: http://localhost:8000/app")
    print("   API Docs: http://localhost:8000/docs\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)