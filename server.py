"""
Launch: python server.py
Opens at: http://localhost:8000/app
"""
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded .env from {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

# Import your existing app (this brings in all API routes)
from main import app

# ==========================================
# CORS — let frontend talk to backend
# ==========================================
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Frontend file paths
# ==========================================
frontend_dir = Path(__file__).parent / "frontend"
css_dir = frontend_dir / "css"
js_dir = frontend_dir / "js"

# Create directories if they don't exist
frontend_dir.mkdir(exist_ok=True)
css_dir.mkdir(exist_ok=True)
js_dir.mkdir(exist_ok=True)

# ==========================================
# Static asset mounts (CSS, JS, images)
# These MUST come before the catch-all routes
# ==========================================
app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")

# If you have an assets folder for images/fonts later:
assets_dir = frontend_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

# ==========================================
# HTML Page Routes
# ==========================================
@app.get("/app", include_in_schema=False)
@app.get("/app/", include_in_schema=False)
async def serve_home():
    """Serve the main home page"""
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/app/citation-auditor", include_in_schema=False)
@app.get("/app/citation-auditor.html", include_in_schema=False)
@app.get("/citation-auditor.html", include_in_schema=False)
async def serve_citation_auditor():
    """Serve the Citation Auditor page"""
    filepath = frontend_dir / "citation-auditor.html"
    if filepath.exists():
        return FileResponse(str(filepath))
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/app/bail-reckoner", include_in_schema=False)
@app.get("/app/bail-reckoner.html", include_in_schema=False)
@app.get("/bail-reckoner.html", include_in_schema=False)
async def serve_bail_reckoner():
    """Serve the Bail Reckoner page"""
    filepath = frontend_dir / "bail-reckoner.html"
    if filepath.exists():
        return FileResponse(str(filepath))
    return FileResponse(str(frontend_dir / "index.html"))


# ==========================================
# Startup banner
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("⚖️  LexAI — Legal Intelligence Platform")
    print("=" * 50)
    print(f"   Frontend:         http://localhost:8000/app")
    print(f"   Citation Auditor: http://localhost:8000/app/citation-auditor")
    print(f"   Bail Reckoner:    http://localhost:8000/app/bail-reckoner")
    print(f"   API Docs:         http://localhost:8000/docs")
    print(f"   API Root:         http://localhost:8000/")
    print("=" * 50)
    print(f"   Frontend dir:     {frontend_dir.resolve()}")
    
    # Verify frontend files exist
    for fname in ["index.html", "citation-auditor.html", "bail-reckoner.html"]:
        fpath = frontend_dir / fname
        status = "✅" if fpath.exists() else "❌ MISSING"
        print(f"   {status}  {fname}")
    for fname in ["css/styles.css", "js/main.js", "js/citation-auditor.js", "js/bail-reckoner.js"]:
        fpath = frontend_dir / fname
        status = "✅" if fpath.exists() else "❌ MISSING"
        print(f"   {status}  {fname}")
    
    print("=" * 50 + "\n")
    
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)