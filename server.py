"""
Launch: python server.py
Opens at: http://localhost:8000/app
"""
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

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