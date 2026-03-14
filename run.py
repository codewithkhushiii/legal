"""
Combined server that serves the FastAPI backend + static frontend files.
Run with: python run.py
"""
import uvicorn
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Import the FastAPI app from your backend
# Adjust the import path based on your actual file structure
import sys
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from main import app  # Your existing FastAPI app from main.py

# Mount the frontend as static files
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    
    # Serve index.html at /app
    from fastapi.responses import FileResponse
    
    @app.get("/app", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
    
    # Also serve at /ui for convenience
    @app.get("/ui", include_in_schema=False)
    async def serve_frontend_alt():
        return FileResponse(str(frontend_dir / "index.html"))

    print(f"✅ Frontend mounted! Access at http://localhost:8000/app")
else:
    print(f"⚠️  Frontend directory not found at {frontend_dir}")

if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend", "frontend"]
    )