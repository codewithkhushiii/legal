# Codebase from `/home/aceninja/shtudy/legal`

### File: `detector.py`

```py
import pandas as pd
import re
import os
import json
from pathlib import Path
from groq import Groq

# 1. Initialize Groq (Use environment variables in production!)
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# ==========================================
# 1. Load ALL metadata into RAM (Same as before)
# ==========================================
print("Gathering metadata from all folders...")
all_dataframes = []

for file_path in Path('.').rglob('*.parquet'):
    if 'venv' in file_path.parts: 
        continue
    try:
        df_part = pd.read_parquet(file_path)
        all_dataframes.append(df_part)
    except Exception as e:
        pass

if all_dataframes:
    df = pd.concat(all_dataframes, ignore_index=True)
    # Give every row a unique ID so Groq can tell us exactly which one it picked
    df = df.reset_index() 
    print("✅ Successfully loaded metadata!")
else:
    print("❌ No valid .parquet files were found.")
    exit()

# ==========================================
# 2. The LLM Verification Engine
# ==========================================

def get_broad_candidates(case_name, max_results=15):
    """Casts a wide net to find potential matches using the most unique word."""
    # Clean the string and split into words
    cleaned = re.sub(r'[^a-zA-Z\s]', '', case_name)
    words = cleaned.split()
    
    # Filter out boring legal words to find the actual name
    stop_words = ['the', 'state', 'union', 'of', 'india', 'vs', 'v', 'versus', 'ors', 'and', 'others', 'anr']
    meaningful_words = [w for w in words if w.lower() not in stop_words]
    
    # Grab the first unique name (e.g., "Vashist", "Divya")
    search_word = meaningful_words[0] if meaningful_words else (words[0] if words else "")
    
    if not search_word:
        return pd.DataFrame() # Empty
        
    # Search dataframe for this specific word
    matches = df[df['title'].str.contains(search_word, case=False, na=False)]
    return matches.head(max_results)


def resolve_match_with_llm(target_citation, candidates_df):
    """Sends the candidates to Groq to pick the true match, strictly."""
    if candidates_df.empty:
        return {"status": "🔴 NO CANDIDATES", "message": "No similar cases found in database."}
        
    print("🧠 Asking Groq to strictly evaluate candidates...")
    
    candidate_dict = {}
    for _, row in candidates_df.iterrows():
        candidate_dict[str(row['index'])] = f"{row['title']} (Year: {row.get('year', 'Unknown')})"
        
    prompt = f"""
    You are a STRICT and UNFORGIVING legal AI auditor. 
    Your job is to verify if a Target Citation exists in the Database Candidates.
    
    Target Citation: "{target_citation}"
    
    Database Candidates:
    {json.dumps(candidate_dict, indent=2)}
    
    STRICT MATCHING RULES:
    1. Minor formatting differences ("v." vs "versus", "& Ors" vs "AND OTHERS") are ALLOWED.
    2. Missing articles (e.g., dropping "The" before "State") are ALLOWED.
    3. DIFFERENT PROPER NOUNS ARE STRICTLY PROHIBITED. "Ajay Kumar Mishra" is NOT "Ajay Krishan Shinghal". If the core surnames or company names do not match exactly, YOU MUST RETURN null.
    4. Do not guess or approximate. If you are not 100% sure the actual parties are identical, return null.
    
    If there is an EXACT match for the core parties, return its ID as a string. 
    If there is NO EXACT MATCH (or if it's a completely different person), return the string "null" for the matched_id.
    
    Respond ONLY with a valid JSON object in this format:
    {{"matched_id": "the_id_string_or_null", "reason": "Short 1 sentence explanation of why it matches perfectly, or why it fails the strict rules."}}
    """

    try:
        response = client.chat.completions.create(
            # Swapped to a smarter model for better reasoning!
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": "You output only valid JSON. You are a strict auditor."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # Keep this at 0 for strict logic
        )
        
        result_json = json.loads(response.choices[0].message.content)
        matched_id = result_json.get("matched_id")
        reason = result_json.get("reason", "No reason provided.")
        
        # Check if the LLM returned our "null" string
        if matched_id and matched_id != "null":
            winning_row = df[df['index'] == int(matched_id)].iloc[0]
            return {
                "status": "🟢 VERIFIED BY AI",
                "matched_name": winning_row['title'],
                "file_to_open": winning_row.get('path', 'Unknown PDF'),
                "reason": reason
            }
        else:
            return {
                "status": "🔴 HALLUCINATION DETECTED",
                "message": f"AI concluded no perfect matches exist. Reason: {reason}"
            }
            
    except Exception as e:
        return {"status": "ERROR", "message": f"Groq API Error: {str(e)}"}
    
    
# ==========================================
# --- Hackathon Demo Tests ---
# ==========================================
cases_to_check = [
    'Vashist Narayan Kumar v. The State of Bihar & Ors.', 
    'Divya vs. Union of India & Ors.', 
    'Ajay Kumar Mishra vs. Union of India & Ors.'
]

print("\n🚀 Starting LLM-Powered Batch Audit...\n")

for case in cases_to_check:
    print(f"--- Auditing: {case} ---")
    
    # 1. Get 15 potential candidates based on the petitioner's name
    candidates = get_broad_candidates(case)
    
    # 2. Let Groq figure out which one is the real match
    result = resolve_match_with_llm(case, candidates)
    
    print(f"Status: {result['status']}")
    if "message" in result:
        print(f"Message: {result['message']}")
    if "matched_name" in result:
        print(f"Matched Case: {result['matched_name']}")
        print(f"PDF Path: {result['file_to_open']}")
        print(f"AI Reasoning: {result.get('reason')}")
        
    print("\n" + "-"*40 + "\n")
```

---

### File: `explore.py`

```py
import pandas as pd
from pathlib import Path

def explore_descriptions():
    print("🔍 Searching for Parquet files...")
    
    # Find all parquet files, ignoring the virtual environment
    parquet_files = [f for f in Path('.').rglob('*.parquet') if 'venv' not in f.parts]

    if not parquet_files:
        print("❌ No parquet files found in the current directory.")
        return

    # Just grab the first file we find for a quick peek
    file_to_load = parquet_files[0]
    print(f"📂 Loading data from: {file_to_load}\n")
    
    try:
        df = pd.read_parquet(file_to_load)
        
        if 'description' not in df.columns:
            print("❌ 'description' column not found in this dataset.")
            print(f"Available columns are: {list(df.columns)}")
            return

        # Drop NaNs or completely empty strings so we don't just print blank spaces
        valid_descriptions = df['description'].dropna()
        valid_descriptions = valid_descriptions[valid_descriptions.str.strip() != ""]
        
        if valid_descriptions.empty:
            print("⚠️ The 'description' column exists, but it's completely empty in this file!")
            return

        print(f"✅ Found {len(valid_descriptions)} valid descriptions. Here are the top 5:\n")
        
        # Grab the top 5 and print them cleanly
        top_5 = valid_descriptions.head(5).tolist()
        
        print(top_5)

        for i, desc in enumerate(top_5, 1):
            print(f"--- 📄 Sample {i} ---")
            # Truncate at 1000 characters just in case it's a massive wall of text
            text_to_print = desc if len(desc) < 1000 else desc[:1000] + "\n...[TRUNCATED]"
            print(text_to_print)
            print("-" * 60 + "\n")

    except Exception as e:
        print(f"⚠️ Error reading file: {e}")

if __name__ == "__main__":
    explore_descriptions()
```

---

### File: `main.py`

```py
import os
import re
import json
import io
import logging
import asyncio
import pandas as pd
import numpy as np
import PyPDF2
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# ==========================================
# 1. Configuration & Global State
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

print("📥 Loading Embedding Model for RAG...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Global variable to hold our database
df = pd.DataFrame()
bail_df = pd.DataFrame() # Your new Reckoner DB

# ==========================================
# 2. Server Startup (Load Data)
# ==========================================
class ReckonerRequest(BaseModel):
    statute: str
    offense_category: str
    imprisonment_duration_served: int
    risk_of_escape: bool
    risk_of_influence: bool
    served_half_term: bool
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, bail_df
    print("🚀 Starting Server: Gathering metadata from all folders...")
    all_dataframes = []

    for file_path in Path('.').rglob('*.parquet'):
        if 'venv' in file_path.parts:
            continue
        try:
            df_part = pd.read_parquet(file_path)
            all_dataframes.append(df_part)
        except Exception as e:
            logger.warning(f"⚠️ Could not load {file_path}: {e}")

    if all_dataframes:
        df = pd.concat(all_dataframes, ignore_index=True)
        df = df.reset_index()
        print(f"✅ Successfully loaded {len(df)} records into RAM!")
        print(f"📋 Columns available: {list(df.columns)}")
        if 'path' in df.columns and len(df) > 0:
            sample_paths = df['path'].dropna().head(5).tolist()
            print(f"📂 Sample 'path' values: {sample_paths}")
    else:
        print("❌ WARNING: No valid .parquet files were found. Search will fail.")

    
    csv_path = Path("a.csv")
    if csv_path.exists():
        try:
            bail_df = pd.read_csv(csv_path)
            print(f"⚖️ Successfully loaded {len(bail_df)} records into the Bail Reckoner!")
        except Exception as e:
            print(f"⚠️ Could not load Bail Reckoner CSV: {e}")
    else:
        print("❌ WARNING: bail_reckoner_database.csv not found.")

    yield
    print("🛑 Shutting down server...")


# Initialize FastAPI
app = FastAPI(title="Legal Citation Auditor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# ==========================================
# 3. Pydantic Models
# ==========================================
class ChatMessage(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    audit_context: Optional[str] = None

class CitationRequest(BaseModel):
    citation: str

class SummaryRequest(BaseModel):
    results: list
    total: int
    sc_count: int
    hc_count: int
    
@app.get("/", response_class=FileResponse)
async def serve_homepage():
    return FileResponse("frontend/templates/index.html")

# Citation Auditor route
@app.get("/auditor", response_class=FileResponse)
async def serve_auditor():
    return FileResponse("frontend/templates/auditor.html")

# Bail Reckoner route
@app.get("/bail-reckoner", response_class=FileResponse)
async def serve_bail_reckoner():
    return FileResponse("frontend/templates/bail-reckoner.html")

# API stats endpoint (keep your existing one, just rename the root)
@app.get("/api/health")
def api_health():
    return {
        "message": "Legal AI Platform API is running!",
        "endpoints": ["/audit-document", "/audit-multiple", "/verify-citation", "/chat", "/summarize", "/reckoner/bail", "/db-stats"],
        "docs": "/docs"
    }

# ==========================================
# 4. Core Helper Functions
# ==========================================
def chunk_text(text, chunk_size=10000, overlap=2000):
    """Splits text into overlapping chunks so citations aren't cut in half."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def smart_chunk_judgment(text, chunk_size=1500, overlap=300):
    """
    Splits a judgment into chunks that preserve paragraph boundaries.
    This is critical because legal judgments have:
      - Party arguments (which may OPPOSE the final holding)
      - Court's analysis
      - The actual holding/ratio decidendi
    
    We need chunks large enough to capture context around each point.
    """
    # First, try to split by paragraph numbers (common in Indian judgments)
    # Pattern: lines starting with numbers like "13.", "14.", etc.
    para_pattern = r'\n\s*(\d{1,3})\.\s+'
    
    # Split by numbered paragraphs
    splits = re.split(para_pattern, text)
    
    paragraphs = []
    i = 0
    while i < len(splits):
        if i + 1 < len(splits) and re.match(r'^\d{1,3}$', splits[i].strip()):
            # Combine paragraph number with its content
            para_text = f"{splits[i].strip()}. {splits[i+1]}"
            paragraphs.append(para_text.strip())
            i += 2
        else:
            if splits[i].strip() and len(splits[i].strip()) > 50:
                paragraphs.append(splits[i].strip())
            i += 1
    
    # If paragraph splitting didn't produce good results, fall back to double-newline
    if len(paragraphs) < 5:
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 80]
    
    # If still not enough, use fixed-size chunking
    if len(paragraphs) < 5:
        paragraphs = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to break at a sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start + chunk_size - 200, end)
                if last_period > start:
                    end = last_period + 1
            paragraphs.append(text[start:end].strip())
            start = end - overlap
    
    # Now merge very small paragraphs with their neighbors
    merged = []
    buffer = ""
    for p in paragraphs:
        if len(buffer) + len(p) < chunk_size:
            buffer = buffer + "\n\n" + p if buffer else p
        else:
            if buffer:
                merged.append(buffer)
            buffer = p
    if buffer:
        merged.append(buffer)
    
    return merged if merged else paragraphs


def _read_pdf(filepath: Path) -> str:
    """Helper function to extract text from a PDF given its path."""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    except Exception as e:
        logger.error(f"⚠️ Error reading PDF {filepath}: {e}")
        return ""


@app.post("/reckoner/bail")
async def calculate_bail_eligibility(req: ReckonerRequest):
    """Bail Reckoner: Calculates risk scores and bail probability based on historical CSV data."""
    if bail_df.empty:
        raise HTTPException(status_code=500, detail="Bail database not loaded.")

    print(f"🧮 Running Bail Reckoner for: {req.statute} - {req.offense_category}")

    # 1. Filter the database for similar historical cases
    # We use exact matches for statute and category to find relevant precedents
    mask = (
        (bail_df['statute'].str.lower() == req.statute.lower()) &
        (bail_df['offense_category'].str.lower() == req.offense_category.lower())
    )
    
    similar_cases = bail_df[mask]

    if similar_cases.empty:
        return JSONResponse({
            "status": "NO_DATA",
            "message": "No historical cases match this specific statute and offense category.",
            "recommendation": "Manual assessment required."
        })

    # 2. Calculate Aggregated Metrics from similar cases
    total_cases = len(similar_cases)
    
    # Bail Probability (Percentage of similar cases that got bail)
    bail_granted = similar_cases['bail_eligibility'].sum()
    bail_probability = (bail_granted / total_cases) * 100

    # Average Risk Score
    avg_risk_score = similar_cases['risk_score'].mean()
    
    # Bond Requirements (Most common requirements for this offense)
    surety_prob = (similar_cases['surety_bond_required'].sum() / total_cases) * 100
    personal_prob = (similar_cases['personal_bond_required'].sum() / total_cases) * 100

    # 3. Apply Hard Logical Rules (e.g., CrPC Section 436A for serving half term)
    statutory_warning = None
    if req.served_half_term and req.statute.lower() != "pmla":
        # Under Sec 436A CrPC, if half the maximum term is served, bail is generally a statutory right
        statutory_warning = "⚠️ Client has served half their term. Under Sec 436A CrPC, they have a strong statutory ground for default bail, overriding standard risk scores."
        bail_probability = max(bail_probability, 90.0) # Boost probability

    # 4. Format the final intelligence report
    # 4. Format the final intelligence report
    return JSONResponse({
        "status": "SUCCESS",
        "inputs_analyzed": {
            "statute": req.statute,
            "category": req.offense_category,
            "time_served": req.imprisonment_duration_served
        },
        "historical_insights": {
            "similar_cases_analyzed": int(total_cases),
            "historical_bail_probability": f"{float(bail_probability):.1f}%",
            "average_risk_score": float(round(avg_risk_score, 2)),
        },
        "likely_conditions": {
            # Wrapped in bool() to fix the JSON serialization error!
            "surety_bond_likely": bool(surety_prob > 50),
            "personal_bond_likely": bool(personal_prob > 50)
        },
        "legal_strategy_note": statutory_warning if statutory_warning else "Standard bail arguments apply. Focus on mitigating flight risk."
    })
def extract_text_from_pdf_path(db_path_value: str) -> str:
    """
    Locate and read a PDF based on the 'path' column from the parquet database.
    
    The 'path' column contains values like '2024_1_1_10'
    The actual file on disk is at:
      judgments_2024/english/extracted_2024_cases/2024_1_1_10_EN.pdf
    """
    if not db_path_value or pd.isna(db_path_value):
        logger.warning("⚠️ No path value provided")
        return ""
    
    path_str = str(db_path_value).strip()
    logger.info(f"🔍 Attempting to locate PDF for path value: '{path_str}'")
    
    # STRATEGY 0: If the path is already a full valid path on disk
    direct_path = Path(path_str)
    if direct_path.exists() and direct_path.suffix == '.pdf':
        logger.info(f"📖 Direct path found: {direct_path}")
        return _read_pdf(direct_path)
    
    if not path_str.lower().endswith('.pdf'):
        direct_with_suffix = Path(path_str + "_EN.pdf")
        if direct_with_suffix.exists():
            logger.info(f"📖 Direct path with suffix found: {direct_with_suffix}")
            return _read_pdf(direct_with_suffix)
    
    # STRATEGY 1: Extract the base file ID and construct the path
    base_name = Path(path_str).stem
    base_name = re.sub(r'_EN$', '', base_name, flags=re.IGNORECASE)
    
    if '/' in base_name or '\\' in base_name:
        base_name = Path(base_name).name
    
    logger.info(f"🔑 Extracted base file ID: '{base_name}'")
    
    year_match = re.match(r'^(\d{4})', base_name)
    if year_match:
        year = year_match.group(1)
        
        for folder_prefix in ["judgments", "judgement", "judgment"]:
            for case_folder in [f"extracted_{year}_cases", f"extracted_{year}_case"]:
                candidate = Path(f"{folder_prefix}_{year}") / "english" / case_folder / f"{base_name}_EN.pdf"
                if candidate.exists():
                    logger.info(f"📖 Successfully located PDF: {candidate}")
                    return _read_pdf(candidate)
    
    # STRATEGY 2: Global search as fallback
    search_pattern = f"{base_name}_EN.pdf"
    logger.info(f"⚠️ Strict path not found. Searching globally for: {search_pattern}")
    
    found = [f for f in Path('.').rglob(search_pattern) if 'venv' not in f.parts]
    if found:
        logger.info(f"📖 Found PDF via global search: {found[0]}")
        return _read_pdf(found[0])
    
    search_pattern_no_en = f"{base_name}.pdf"
    found2 = [f for f in Path('.').rglob(search_pattern_no_en) if 'venv' not in f.parts]
    if found2:
        logger.info(f"📖 Found PDF (no _EN suffix): {found2[0]}")
        return _read_pdf(found2[0])
    
    logger.warning(f"❌ File completely missing from disk: {search_pattern}")
    return ""


def verify_quotation(attributed_claim: str, source_file_path: str):
    """
    The RAG Engine: Finds the relevant context in the judgment and uses LLM to verify.
    
    KEY IMPROVEMENTS:
    1. Uses smart paragraph-aware chunking instead of naive splitting
    2. Retrieves TOP-K chunks (not just top-1) to capture full context
    3. Specifically instructs the LLM about how legal judgments work
       (party arguments vs court's own holding)
    """
    if not attributed_claim or not attributed_claim.strip():
        return {"status": "⚠️ SKIPPED", "reason": "No specific claim was extracted to verify."}

    source_text = extract_text_from_pdf_path(source_file_path)
    
    if not source_text.strip():
        return {
            "status": "⚠️ ERROR",
            "reason": f"Could not extract text from the source judgment. (path: {source_file_path})"
        }

    # ─────────────────────────────────────────────
    # STEP 1: Smart chunking that preserves paragraph structure
    # ─────────────────────────────────────────────
    paragraphs = smart_chunk_judgment(source_text, chunk_size=1500, overlap=300)

    if not paragraphs:
        return {"status": "⚠️ ERROR", "reason": "No valid text chunks found in source."}

    logger.info(f"🧠 Embedding {len(paragraphs)} smart chunks to find relevant context...")
    
    # ─────────────────────────────────────────────
    # STEP 2: Retrieve TOP-K most relevant chunks (not just top-1)
    # ─────────────────────────────────────────────
    claim_embedding = embedder.encode([attributed_claim])
    para_embeddings = embedder.encode(paragraphs)
    similarities = cosine_similarity(claim_embedding, para_embeddings)[0]
    
    # Get top 5 most relevant chunks
    TOP_K = 5
    top_indices = np.argsort(similarities)[-TOP_K:][::-1]  # Descending order
    
    max_score = similarities[top_indices[0]]
    
    if max_score < 0.20:
        return {
            "status": "🟡 UNVERIFIABLE",
            "reason": f"Could not find sufficiently relevant content in the judgment. (Max similarity: {max_score:.2f}). The claim may use different terminology than the judgment.",
            "closest_text_found": paragraphs[top_indices[0]][:300] + "..."
        }

    # ─────────────────────────────────────────────
    # STEP 3: Combine top chunks to give LLM full context
    # ─────────────────────────────────────────────
    # Sort the top chunks by their original position in the document
    # This preserves the narrative flow (arguments → analysis → holding)
    sorted_indices = sorted(top_indices, key=lambda x: x)
    
    combined_context = ""
    for idx in sorted_indices:
        score = similarities[idx]
        if score >= 0.15:  # Only include reasonably relevant chunks
            combined_context += f"\n\n[--- Relevant Section (similarity: {score:.2f}) ---]\n"
            combined_context += paragraphs[idx]
    
    # Also try to include the FINAL paragraphs of the judgment 
    # (where the actual order/holding usually is)
    total_paras = len(paragraphs)
    if total_paras > 3:
        final_section = "\n\n".join(paragraphs[-3:])
        # Only add if not already included
        if paragraphs[-1] not in combined_context:
            combined_context += f"\n\n[--- Final Section of Judgment ---]\n{final_section}"

    # ─────────────────────────────────────────────
    # STEP 4: Improved LLM verification prompt
    # ─────────────────────────────────────────────
    verification_prompt = f"""You are a SENIOR legal analyst verifying whether a claim attributed to a court judgment is accurate.

CRITICAL LEGAL CONTEXT:
Indian court judgments typically have this structure:
1. Facts of the case
2. Arguments by the petitioner/appellant  
3. Arguments by the respondent/state
4. The COURT'S OWN ANALYSIS and reasoning
5. The FINAL HOLDING/ORDER

IMPORTANT: A paragraph may describe what one PARTY argued, or what a LOWER COURT held, which the Supreme Court may have OVERRULED or DISAGREED with. 
Do NOT confuse a party's argument or a lower court's ruling with the SUPREME COURT'S OWN HOLDING.

The claim being verified is:
"{attributed_claim}"

Here are the most relevant excerpts from the actual judgment (in document order):
{combined_context}

YOUR TASK:
1. First, identify whether the relevant excerpts show the COURT'S OWN VIEW or merely describe a party's argument / lower court ruling
2. Then determine: Does the court's ACTUAL HOLDING/RATIO support this claim?

Consider these possibilities:
- SUPPORTED: The court's own analysis/holding supports or is consistent with the claimed principle
- CONTRADICTED: The court explicitly held the OPPOSITE of what is claimed  
- UNSUPPORTED: The judgment discusses related topics but doesn't clearly establish the claimed principle
- PARTIALLY_SUPPORTED: The claim captures the general spirit but oversimplifies or slightly misrepresents the holding

Respond ONLY in JSON format:
{{"verdict": "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED" | "PARTIALLY_SUPPORTED", "explanation": "2-3 sentence reasoning explaining which part of the judgment you relied on and whether it represents the court's own view or a party's argument"}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a senior legal analyst specializing in Indian Supreme Court jurisprudence. You understand the difference between obiter dicta, ratio decidendi, party arguments, and the court's own holding. You are careful and nuanced in your analysis."
                },
                {"role": "user", "content": verification_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        
        status_map = {
            "SUPPORTED": "🟢 QUOTE VERIFIED",
            "PARTIALLY_SUPPORTED": "🟡 PARTIALLY VERIFIED",
            "CONTRADICTED": "🔴 QUOTE CONTRADICTED",
            "UNSUPPORTED": "🟡 QUOTE UNVERIFIABLE"
        }
        
        verdict = result.get("verdict", "UNSUPPORTED")
        
        return {
            "status": status_map.get(verdict, "⚠️ UNKNOWN"),
            "verdict": verdict,
            "reason": result.get("explanation", "No explanation provided."),
            "similarity_score": float(max_score),
            "chunks_analyzed": len([i for i in sorted_indices if similarities[i] >= 0.15]),
            "found_paragraph": paragraphs[top_indices[0]]  # Still include the best matching chunk
        }
    except Exception as e:
        return {"status": "⚠️ ERROR", "reason": f"LLM Error: {str(e)}"}


def extract_citations_with_groq(full_text):
    print("🧠 Splitting document into manageable chunks...")
    chunks = chunk_text(full_text)
    
    all_extracted_cases = {}

    for i, chunk in enumerate(chunks):
        prompt = f"""
        You are an expert legal AI auditor. Read this section of a legal document.
        Extract EVERY single legal case, precedent, or judgment mentioned.
        
        For each case, classify the court based on context:
        - If it mentions "Del", "Bom", "Mad", or "High Court", classify as "High Court".
        - If it mentions "INSC", "SCC", or implies the apex court, classify as "Supreme Court".
        - If you cannot tell, classify as "Unknown".
        
        ALSO, extract the exact claim, reasoning, or principle attributed to this case by the author.
        Be precise: extract what the AUTHOR SAYS the court held, not background facts.
        
        Respond ONLY with a valid JSON object in this exact format:
        {{
          "citations": [
            {{
              "case_name": "Exact Name of Case",
              "court_type": "Supreme Court",
              "attributed_claim": "The court held that X equals Y"
            }}
          ]
        }}
        
        Document Text Chunk:
        {chunk}
        """

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a strict data extractor that outputs valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            result_json = json.loads(response.choices[0].message.content)
            extracted_in_chunk = result_json.get("citations", [])

            for item in extracted_in_chunk:
                name = item.get("case_name")
                if name and name not in all_extracted_cases:
                    all_extracted_cases[name] = item

        except Exception as e:
            logger.error(f"❌ Error extracting citations from chunk {i+1}: {e}")

    supreme_court_cases = []
    high_court_cases = []

    for name, item in all_extracted_cases.items():
        if item.get("court_type") == "High Court":
            high_court_cases.append(name)
        else:
            supreme_court_cases.append(name)

    return {
        "sc_cases": supreme_court_cases,
        "hc_cases": high_court_cases,
        "details": all_extracted_cases
    }


def get_broad_candidates(case_name, max_results=15):
    if df.empty:
        return pd.DataFrame()

    query = str(case_name).strip()
    
    # STRATEGY 1: Neutral Citation or Number Match
    if any(char.isdigit() for char in query):
        escaped_query = re.escape(query)
        mask = (
            df.get('nc_display', pd.Series(dtype=str)).astype(str).str.contains(escaped_query, case=False, na=False) |
            df.get('citation', pd.Series(dtype=str)).astype(str).str.contains(escaped_query, case=False, na=False) |
            df['title'].astype(str).str.contains(escaped_query, case=False, na=False)
        )
        matches = df[mask]
        if not matches.empty:
            return matches.head(max_results)

    # STRATEGY 2: Text/Name Match (Fallback)
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', query)
    words = cleaned.split()
    stop_words = ['the', 'state', 'union', 'of', 'india', 'vs', 'v', 'versus', 'ors', 'and', 'others', 'anr', 'insc', 'scc']
    meaningful_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    
    if not meaningful_words:
        return pd.DataFrame()

    search_word = max(meaningful_words, key=len)
    escaped_search = re.escape(search_word)
    matches = df[df['title'].astype(str).str.contains(escaped_search, case=False, na=False)]
    
    if (matches.empty or len(matches) > 50) and len(meaningful_words) >= 2:
        sorted_words = sorted(meaningful_words, key=len, reverse=True)[:2]
        combined_mask = pd.Series([True] * len(df))
        for w in sorted_words:
            combined_mask = combined_mask & df['title'].astype(str).str.contains(re.escape(w), case=False, na=False)
        combined_matches = df[combined_mask]
        if not combined_matches.empty:
            return combined_matches.head(max_results)
    
    return matches.head(max_results)


def resolve_match_with_llm(target_citation, candidates_df):
    if candidates_df.empty:
        return {"status": "🔴 NO CANDIDATES", "message": "No similar cases found in database.", "confidence": 0}

    candidate_dict = {
        str(row['index']): f"{row.get('title', 'Unknown')} (NC: {row.get('nc_display', 'N/A')}, Citation: {row.get('citation', 'N/A')})"
        for _, row in candidates_df.iterrows()
    }

    prompt = f"""
    You are a STRICT legal AI auditor. Verify if a Target Citation exists in the Database Candidates.
    Target Citation: "{target_citation}"
    Database Candidates: {json.dumps(candidate_dict)}
    
    RULES:
    1. Minor formatting differences ("v." vs "versus") or missing articles ("The") are ALLOWED.
    2. If the neutral citation (NC) matches (e.g. 2024 INSC 2), it is an EXACT match.
    3. If party names match on both sides (petitioner vs respondent), it is an EXACT match.
    If there is an EXACT match, return its ID. If NO EXACT MATCH, return "null".
    Respond ONLY with JSON: {{"matched_id": "id_or_null", "reason": "Short reason", "confidence": 85}}
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You output only valid JSON. You are a strict auditor."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result_json = json.loads(response.choices[0].message.content)
        matched_id = result_json.get("matched_id")
        reason = result_json.get("reason", "No reason provided.")
        confidence = result_json.get("confidence", 50)

        if matched_id and matched_id != "null":
            try:
                winning_row = df[df['index'] == int(matched_id)].iloc[0]
            except (ValueError, IndexError) as e:
                logger.error(f"Could not find matched_id {matched_id} in dataframe: {e}")
                return {
                    "status": "🔴 HALLUCINATION DETECTED",
                    "message": f"LLM returned invalid ID: {matched_id}",
                    "confidence": 0
                }
            
            file_path_val = winning_row.get('path', '')
            
            return {
                "status": "🟢 VERIFIED BY AI",
                "matched_name": winning_row.get('title', 'Unknown'),
                "matched_citation": winning_row.get('nc_display', winning_row.get('citation', 'N/A')),
                "file_to_open": str(file_path_val) if pd.notna(file_path_val) else "",
                "reason": reason,
                "confidence": confidence
            }
        return {
            "status": "🔴 HALLUCINATION DETECTED",
            "message": f"Reason: {reason}",
            "confidence": confidence
        }
    except Exception as e:
        return {"status": "ERROR", "message": f"Groq API Error: {str(e)}", "confidence": 0}


# ==========================================
# 5. API Endpoints
# ==========================================

@app.get("/")
def read_root():
    return {
        "message": "Legal Citation Auditor API is running!",
        "endpoints": ["/audit-document", "/audit-multiple", "/verify-citation", "/chat", "/summarize", "/db-stats"],
        "docs": "/docs"
    }

@app.get("/db-stats")
def get_db_stats():
    if df.empty:
        return {"loaded": False, "record_count": 0}
    
    sample_paths = []
    if 'path' in df.columns:
        sample_paths = df['path'].dropna().head(3).tolist()
    
    return {
        "loaded": True,
        "record_count": len(df),
        "columns": list(df.columns),
        "sample_paths": sample_paths
    }

@app.post("/audit-document")
async def audit_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        file_bytes = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        document_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

    if not document_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    extracted_data = extract_citations_with_groq(document_text)
    sc_citations = extracted_data["sc_cases"]
    hc_citations = extracted_data["hc_cases"]
    case_details = extracted_data["details"]

    if not sc_citations and not hc_citations:
        return JSONResponse({"message": "No citations found in the document.", "results": []})

    final_report = []

    for citation in sc_citations:
        candidates = get_broad_candidates(citation)
        verification_result = resolve_match_with_llm(citation, candidates)
        
        quote_verification = {}
        claim = case_details.get(citation, {}).get("attributed_claim", "")
        
        if "🟢" in verification_result.get("status", ""):
            source_file_path = verification_result.get("file_to_open", "")
            if not source_file_path or source_file_path.strip() == "":
                quote_verification = {"status": "⚠️ SKIPPED", "reason": "No PDF path available in database for this case."}
            else:
                quote_verification = verify_quotation(claim, source_file_path)
        else:
            quote_verification = {"status": "⚠️ SKIPPED", "reason": "Case was not verified, skipping quote check."}

        final_report.append({
            "target_citation": citation,
            "court_type": "Supreme Court / Unknown",
            "verification": verification_result,
            "quote_verification": quote_verification
        })

    for citation in hc_citations:
        candidates = get_broad_candidates(citation)
        if not candidates.empty:
            verification_result = resolve_match_with_llm(citation, candidates)
            verification_result["status"] = "⚠️ HC-" + verification_result["status"].replace("🟢 ", "").replace("🔴 ", "")
        else:
            verification_result = {
                "status": "⚠️ SKIPPED",
                "message": "Identified as a High Court case. Not verified against the Supreme Court registry.",
                "confidence": 0
            }

        final_report.append({
            "target_citation": citation,
            "court_type": "High Court",
            "verification": verification_result,
            "quote_verification": {"status": "⚠️ SKIPPED", "reason": "High Court quote RAG bypass."}
        })

    return JSONResponse({
        "filename": file.filename,
        "total_citations_found": len(sc_citations) + len(hc_citations),
        "supreme_court_count": len(sc_citations),
        "high_court_count": len(hc_citations),
        "results": final_report
    })

@app.post("/audit-multiple")
async def audit_multiple(files: List[UploadFile] = File(...)):
    all_results = []
    total_sc = 0
    total_hc = 0

    for file in files:
        if not file.filename.endswith('.pdf'):
            all_results.append({"filename": file.filename, "error": "Not a PDF file", "results": []})
            continue

        try:
            file_bytes = await file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            document_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
        except Exception as e:
            all_results.append({"filename": file.filename, "error": f"Error reading PDF: {str(e)}", "results": []})
            continue

        if not document_text.strip():
            all_results.append({"filename": file.filename, "error": "Could not extract text from PDF", "results": []})
            continue

        extracted_data = extract_citations_with_groq(document_text)
        sc_citations = extracted_data["sc_cases"]
        hc_citations = extracted_data["hc_cases"]
        case_details = extracted_data["details"]
        total_sc += len(sc_citations)
        total_hc += len(hc_citations)

        file_report = []
        for citation in sc_citations:
            candidates = get_broad_candidates(citation)
            verification_result = resolve_match_with_llm(citation, candidates)
            
            quote_verification = {}
            claim = case_details.get(citation, {}).get("attributed_claim", "")
            
            if "🟢" in verification_result.get("status", ""):
                source_file_path = verification_result.get("file_to_open", "")
                if source_file_path and source_file_path.strip():
                    quote_verification = verify_quotation(claim, source_file_path)
                else:
                    quote_verification = {"status": "⚠️ SKIPPED", "reason": "No PDF path available."}
            else:
                quote_verification = {"status": "⚠️ SKIPPED", "reason": "Case not verified."}
            
            file_report.append({
                "target_citation": citation,
                "court_type": "Supreme Court / Unknown",
                "verification": verification_result,
                "quote_verification": quote_verification
            })

        for citation in hc_citations:
            file_report.append({
                "target_citation": citation,
                "court_type": "High Court",
                "verification": {"status": "⚠️ SKIPPED", "message": "High Court case bypassed.", "confidence": 0},
                "quote_verification": {"status": "⚠️ SKIPPED", "reason": "High Court quote RAG bypass."}
            })

        all_results.append({
            "filename": file.filename,
            "citations_found": len(sc_citations) + len(hc_citations),
            "sc_count": len(sc_citations),
            "hc_count": len(hc_citations),
            "results": file_report
        })

    verified = sum(1 for doc in all_results for r in doc.get("results", []) if "🟢" in (r.get("verification") or {}).get("status", ""))
    fabricated = sum(1 for doc in all_results for r in doc.get("results", []) if "HALLUCINATION" in (r.get("verification") or {}).get("status", ""))

    return JSONResponse({
        "total_documents": len(files),
        "total_sc_citations": total_sc,
        "total_hc_citations": total_hc,
        "total_verified": verified,
        "total_fabricated": fabricated,
        "documents": all_results
    })

@app.post("/verify-citation")
async def verify_single_citation(req: CitationRequest):
    citation = req.citation.strip()
    if not citation:
        raise HTTPException(status_code=400, detail="Citation text cannot be empty.")

    candidates = get_broad_candidates(citation)
    result = resolve_match_with_llm(citation, candidates)

    hc_keywords = ['high court', ' hc ', 'del hc', 'bom hc', 'mad hc', 'cal hc']
    court_type = "High Court" if any(k in citation.lower() for k in hc_keywords) else "Supreme Court / Unknown"

    return JSONResponse({
        "target_citation": citation,
        "court_type": court_type,
        "verification": result
    })

@app.post("/chat")
async def legal_chat(payload: ChatMessage):
    system_prompt = """You are LexAI, an expert AI legal assistant specializing in Indian law, particularly Supreme Court and High Court jurisprudence.

You help lawyers, legal researchers, and students by:
- Explaining legal concepts, sections, and acts
- Summarizing cases and precedents
- Analyzing audit results from the Citation Auditor tool
- Answering questions about Indian constitutional law, IPC, CrPC, and civil procedure
- Helping identify potential legal arguments and precedents

Always be precise, cite relevant law where possible, and note when you are uncertain.
If the user shares audit results, use them as context to give tailored advice.
Keep responses concise but thorough. Format with bullet points when listing items."""

    messages = [{"role": "system", "content": system_prompt}]

    if payload.audit_context:
        messages.append({"role": "system", "content": f"Current audit context:\n{payload.audit_context}"})

    if payload.history:
        for msg in payload.history[-10:]:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": payload.message})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        return JSONResponse({"reply": response.choices[0].message.content})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.post("/summarize")
async def generate_summary(req: SummaryRequest):
    verified = [r for r in req.results if "🟢" in (r.get("verification") or {}).get("status", "")]
    fabricated = [r for r in req.results if "HALLUCINATION" in (r.get("verification") or {}).get("status", "")]
    skipped = [r for r in req.results if "SKIPPED" in (r.get("verification") or {}).get("status", "")]
    unverified = [r for r in req.results if r not in verified and r not in fabricated and r not in skipped]

    fabricated_names = [r.get("target_citation", "Unknown") for r in fabricated[:5]]

    prompt = f"""You are a senior legal analyst generating an audit report summary.

Audit Data:
- Total citations reviewed: {req.total}
- Supreme Court citations: {req.sc_count}
- High Court citations: {req.hc_count}
- Verified/Upheld: {len(verified)}
- Fabricated/Hallucinated: {len(fabricated)}
- High Court (skipped): {len(skipped)}
- Unverified: {len(unverified)}
- Fabricated citations (up to 5): {', '.join(fabricated_names) if fabricated_names else 'None'}

Write a concise 3-4 paragraph professional legal audit summary in plain English. Include:
1. Overall integrity assessment (is the document trustworthy?)
2. Key findings (which citations were problematic)
3. Risk level (Low/Medium/High) and recommendation for the lawyer

Be direct and professional. Do not use markdown headers."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a senior legal analyst. Write professional audit summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        risk = "High" if len(fabricated) > 2 else ("Medium" if len(fabricated) > 0 else "Low")
        return JSONResponse({
            "summary": response.choices[0].message.content,
            "risk_level": risk,
            "stats": {
                "verified": len(verified),
                "fabricated": len(fabricated),
                "skipped": len(skipped),
                "unverified": len(unverified)
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")
```

---

### File: `update_db.py`

```py
import pandas as pd
import re
from pathlib import Path

def extract_legal_provisions(text):
    """Hunts for Acts and Sections in messy OCR text."""
    if not isinstance(text, str) or not text.strip():
        return ""
        
    provisions = set()
    
    # 1. Catch "Sections X to Y" or "Section X"
    section_pattern = r'(?:Section|Sections|Sec\.)\s*(\d+(?:\s*to\s*\d+)?|[A-Z\d]+)'
    for match in re.findall(section_pattern, text, re.IGNORECASE):
        provisions.add(f"Section {match.strip()}")
        
    # 2. Catch "[Name] Act, [Year]"
    act_pattern = r'([A-Z][A-Za-z\s]+Act,\s*\d{4})'
    for match in re.findall(act_pattern, text):
        provisions.add(match.strip())
        
    # 3. Catch common abbreviations like IPC, CrPC, CPC
    if re.search(r'\bIPC\b', text): provisions.add("IPC")
    if re.search(r'\bCrPC\b', text): provisions.add("CrPC")
    if re.search(r'\bCPC\b', text): provisions.add("CPC")
        
    return ", ".join(sorted(list(provisions)))

def upgrade_all_parquets():
    print("🚀 Starting Database Upgrade: Mining provisions...")
    
    # Find all parquet files, ignoring the venv
    parquet_files = [f for f in Path('.').rglob('*.parquet') if 'venv' not in f.parts]
    
    if not parquet_files:
        print("❌ No .parquet files found.")
        return
        
    total_files = len(parquet_files)
    cases_upgraded = 0
    
    for i, file_path in enumerate(parquet_files, 1):
        print(f"📦 Processing file {i}/{total_files}: {file_path}")
        try:
            df = pd.read_parquet(file_path)
            
            if 'description' not in df.columns:
                print(f"   ⚠️ Skipping (no description column)")
                continue
                
            # Apply our regex hunter to the description column
            df['provisions'] = df['description'].apply(extract_legal_provisions)
            
            # Save it right back to the same file
            df.to_parquet(file_path)
            cases_upgraded += len(df)
            print(f"   ✅ Upgraded {len(df)} cases.")
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")

    print(f"\n🎉 SUCCESS! Upgraded {cases_upgraded} total cases with a new 'provisions' column.")

if __name__ == "__main__":
    upgrade_all_parquets()
```

---

### File: `walk.py`

```py
import os

def combine_code_to_markdown(repo_path, output_file):
    # The extensions you want to target
    target_extensions = {'.js', '.py', '.css', '.html'}
    
    # Folders to ignore so the script doesn't read massive dependency files
    ignore_folders = {'.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build', '.next'}

    with open(output_file, 'w', encoding='utf-8') as md:
        md.write(f"# Codebase from `{os.path.abspath(repo_path)}`\n\n")

        # Walk through the directory
        for root, dirs, files in os.walk(repo_path):
            # Exclude ignored folders from the search
            dirs[:] = [d for d in dirs if d not in ignore_folders]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                
                # If the file matches our target extensions
                if ext in target_extensions:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    # Remove the dot for the markdown code block identifier (e.g., .js -> js)
                    lang = ext[1:] 

                    # Write the header and open the code block
                    md.write(f"### File: `{rel_path}`\n\n")
                    md.write(f"```{lang}\n")

                    # Read and write the file contents
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            md.write(f.read())
                    except Exception as e:
                        md.write(f"// Error reading file: {e}")

                    # Close the code block
                    md.write("\n```\n\n---\n\n")

if __name__ == "__main__":
    # Settings: '.' means current directory, or you can paste a specific folder path
    repo_directory = "." 
    output_markdown_file = "my_repository_code.md"

    print(f"Scanning '{repo_directory}'...")
    combine_code_to_markdown(repo_directory, output_markdown_file)
    print(f"Done! Your code has been saved to '{output_markdown_file}'.")
```

---

### File: `server.py`

```py
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
```

---

### File: `run.py`

```py
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
```

---

### File: `miner.py`

```py
import pandas as pd
import PyPDF2
import re
import os
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress PyPDF2 warnings about messy PDFs
warnings.filterwarnings("ignore", category=UserWarning)

def extract_legal_provisions(text):
    """Hunts for Acts and Sections in the raw PDF text."""
    if not isinstance(text, str) or not text.strip():
        return ""
        
    provisions = set()
    
    # 1. Catch "Sections X to Y" or "Section X"
    section_pattern = r'(?:Section|Sections|Sec\.)\s*(\d+(?:\s*to\s*\d+)?|[A-Z\d]+)'
    for match in re.findall(section_pattern, text, re.IGNORECASE):
        provisions.add(f"Section {match.strip()}")
        
    # 2. Catch "[Name] Act, [Year]"
    act_pattern = r'([A-Z][A-Za-z\s]+Act,\s*\d{4})'
    for match in re.findall(act_pattern, text):
        provisions.add(match.strip())
        
    # 3. Catch common abbreviations
    if re.search(r'\bIPC\b', text): provisions.add("IPC")
    if re.search(r'\bCrPC\b', text): provisions.add("CrPC")
    if re.search(r'\bCPC\b', text): provisions.add("CPC")
        
    return ", ".join(sorted(list(provisions)))

def find_pdf_path(row):
    """Attempts to locate the PDF file on disk using multiple strategies."""
    # Strategy 1: The 'path' column
    if 'path' in row and pd.notna(row['path']):
        candidate = Path(str(row['path']))
        if candidate.exists():
            return candidate

    # Strategy 2: The 'case_id' or 'nc_display' column (e.g. 2024_1_1_10)
    file_id = None
    if 'nc_display' in row and pd.notna(row['nc_display']):
        file_id = str(row['nc_display']).strip()
    elif 'case_id' in row and pd.notna(row['case_id']):
        file_id = str(row['case_id']).strip()

    if file_id:
        year_match = re.match(r'^(\d{4})', file_id)
        if year_match:
            year = year_match.group(1)
            possible_folders = [f"judgement_{year}", f"judgment_{year}"]
            sub_folder = f"extracted_{year}_case"
            
            for base in possible_folders:
                candidate = Path(base) / sub_folder / f"{file_id}.pdf"
                if candidate.exists():
                    return candidate
    return None

def process_single_pdf(pdf_path, max_pages=3):
    """Reads ONLY the first few pages of a PDF to save extreme amounts of time."""
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
        extracted_text = ""
        # Read up to max_pages, or the length of the document, whichever is smaller
        pages_to_read = min(max_pages, len(reader.pages))
        
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                extracted_text += page_text + "\n"
                
        return extract_legal_provisions(extracted_text)
    except Exception:
        # Silently fail on corrupted PDFs so the script doesn't crash
        return ""

def mine_all_pdfs():
    print("🚀 Starting Deep PDF Mining Operation...")
    
    parquet_files = [f for f in Path('.').rglob('*.parquet') if 'venv' not in f.parts]
    
    if not parquet_files:
        print("❌ No Parquet files found.")
        return

    total_provisions_found = 0

    for file_to_load in parquet_files:
        print(f"\n📂 Loading: {file_to_load}")
        try:
            df = pd.read_parquet(file_to_load)
            
            # Create the column if it doesn't exist yet
            if 'provisions' not in df.columns:
                df['provisions'] = ""

            # Use tqdm to show a progress bar for the rows in THIS specific file
            tqdm.pandas(desc="Mining PDFs")
            
            # Apply our massive extractor function row by row
            def process_row(row):
                # If we already found provisions previously, skip it to save time
                if pd.notna(row.get('provisions')) and str(row.get('provisions')).strip():
                    return row['provisions']
                    
                pdf_path = find_pdf_path(row)
                if pdf_path:
                    return process_single_pdf(pdf_path, max_pages=3)
                return ""

            df['provisions'] = df.progress_apply(process_row, axis=1)
            
            # Count how many we actually found in this batch
            found_count = df['provisions'].apply(lambda x: 1 if str(x).strip() else 0).sum()
            total_provisions_found += found_count
            
            # Overwrite the parquet file with the newly enriched data
            df.to_parquet(file_to_load)
            print(f"✅ Saved. {found_count} cases now have legal provisions in this file.")

        except Exception as e:
            print(f"⚠️ Error processing file {file_to_load}: {e}")

    print("\n" + "="*50)
    print("🎉 MINING COMPLETE!")
    print(f"Total cases enriched with provisions: {total_provisions_found}")
    print("="*50)

if __name__ == "__main__":
    mine_all_pdfs()
```

---

### File: `verify_db.py`

```py
import pandas as pd
from pathlib import Path

def count_descriptions():
    print("🔍 Scanning all Parquet files to count descriptions...")
    
    parquet_files = [f for f in Path('.').rglob('*.parquet') if 'venv' not in f.parts]

    if not parquet_files:
        print("❌ No parquet files found.")
        return

    total_rows = 0
    total_with_desc = 0
    total_without_desc = 0
    files_processed = 0

    for file_to_load in parquet_files:
        try:
            df = pd.read_parquet(file_to_load)
            files_processed += 1
            
            # Count total rows in this file
            current_rows = len(df)
            total_rows += current_rows

            if 'description' not in df.columns:
                # If the column doesn't even exist, all rows are missing descriptions
                total_without_desc += current_rows
                continue

            # Convert to string and strip whitespace to catch sneaky empty strings
            # Replace NaNs with empty strings first
            desc_series = df['description'].fillna("").astype(str).str.strip()
            
            # Count rows where description is NOT empty and NOT "None Found" (based on your log)
            has_desc_mask = (desc_series != "") & (desc_series.str.lower() != "none found")
            
            with_desc = has_desc_mask.sum()
            without_desc = current_rows - with_desc

            total_with_desc += with_desc
            total_without_desc += without_desc

        except Exception as e:
            print(f"⚠️ Error reading {file_to_load}: {e}")

    # Print Final Report
    print("\n" + "="*50)
    print("📊 DATABASE DESCRIPTION REPORT")
    print("="*50)
    print(f"📁 Files Processed      : {files_processed}")
    print(f"📝 Total Cases (Rows)   : {total_rows:,}")
    print(f"✅ With Descriptions    : {total_with_desc:,} ({total_with_desc/total_rows*100:.1f}%)")
    print(f"❌ Without Descriptions : {total_without_desc:,} ({total_without_desc/total_rows*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    count_descriptions()
```

---

### File: `frontend/app.js`

```js
// ==========================================
// ⚖️ LEGAL CITATION AUDITOR v2.1 — ENGINE
// Features: Audit, Search, Bulk, History, Export, Chatbot, Summary, RAG Quote Verification
// ==========================================

const API_BASE = '';

// ===== DOM REFS — AUDIT =====
const oathScreen       = document.getElementById('oath-screen');
const appEl            = document.getElementById('app');
const sealDropzone     = document.getElementById('seal-dropzone');
const fileInput        = document.getElementById('file-input');
const filedDoc         = document.getElementById('filed-doc');
const filedName        = document.getElementById('filed-name');
const filedSize        = document.getElementById('filed-size');
const filedRemove      = document.getElementById('filed-remove');
const gavelBtn         = document.getElementById('gavel-btn');
const chamberIdle      = document.getElementById('chamber-idle');
const chamberDelib     = document.getElementById('chamber-deliberation');
const chamberResults   = document.getElementById('chamber-results');
const judgmentRoll     = document.getElementById('judgment-roll');
const orderOverlay     = document.getElementById('order-overlay');
const orderBody        = document.getElementById('order-body');
const orderClose       = document.getElementById('order-close');
const benchClock       = document.getElementById('bench-clock');
const sessionIdEl      = document.getElementById('session-id');
const toastContainer   = document.getElementById('toast-container');

// Stats
const vcTotal      = document.getElementById('vc-total');
const vcUpheld     = document.getElementById('vc-upheld');
const vcOverruled  = document.getElementById('vc-overruled');
const vcSkipped    = document.getElementById('vc-skipped');
const vcUnheard    = document.getElementById('vc-unheard');

// Quote verification stats
const vcQuoteVerified     = document.getElementById('vc-quote-verified');
const vcQuoteContradicted = document.getElementById('vc-quote-contradicted');
const vcQuoteUnsupported  = document.getElementById('vc-quote-unsupported');
const vcQuoteFabricated   = document.getElementById('vc-quote-fabricated');

// Filter counts
const jfAll        = document.getElementById('jf-all');
const jfUpheld     = document.getElementById('jf-upheld');
const jfFabricated = document.getElementById('jf-fabricated');
const jfSkipped    = document.getElementById('jf-skipped');
const jfUnverified = document.getElementById('jf-unverified');
const jfQuoteIssues = document.getElementById('jf-quote-issues');

// Court breakdown
const cbScFill     = document.getElementById('cb-sc-fill');
const cbHcFill     = document.getElementById('cb-hc-fill');
const cbScCount    = document.getElementById('cb-sc-count');
const cbHcCount    = document.getElementById('cb-hc-count');
const courtBreakdown = document.getElementById('court-breakdown');

let selectedFile = null;
let auditData    = [];
let lastAuditResponse = null;

// ==========================================
// 1. OATH SCREEN (Boot)
// ==========================================
function runOathSequence() {
    // Check backend health during boot
    checkBackendHealth();
    setTimeout(() => {
        oathScreen.classList.add('dismissed');
        appEl.classList.remove('hidden');
        setTimeout(() => { oathScreen.style.display = 'none'; }, 1000);
    }, 4500);
}

async function checkBackendHealth() {
    try {
        const resp = await fetch(`${API_BASE}/db-stats`);
        const data = await resp.json();
        if (data.loaded) {
            document.getElementById('registry-records').textContent = `${data.record_count.toLocaleString()} cases in registry`;
        } else {
            document.getElementById('ind-db').classList.remove('online');
            document.getElementById('ind-db').classList.add('offline');
        }
    } catch (e) {
        console.warn('Backend not reachable:', e);
        ['ind-api', 'ind-llm', 'ind-db', 'ind-rag'].forEach(id => {
            const el = document.getElementById(id);
            if (el) { el.classList.remove('online'); el.classList.add('offline'); }
        });
    }
}

// ==========================================
// 2. CLOCK
// ==========================================
function updateClock() {
    const now = new Date();
    const opts = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
    const dateStr = now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
    benchClock.textContent = `${dateStr}  ${now.toLocaleTimeString('en-IN', opts)}`;
}
setInterval(updateClock, 1000);
updateClock();

sessionIdEl.textContent = `SCI-${Date.now().toString(36).toUpperCase().slice(-6)}`;

// ==========================================
// 3. TAB NAVIGATION
// ==========================================
function switchTab(tab) {
    ['audit', 'search', 'bulk', 'history'].forEach(t => {
        const tabEl = document.getElementById(`tab-${t}`);
        const navEl = document.getElementById(`nav-${t}`);
        if (t === tab) {
            tabEl.classList.remove('hidden');
            navEl.classList.add('active');
        } else {
            tabEl.classList.add('hidden');
            navEl.classList.remove('active');
        }
    });
    if (tab === 'history') renderHistory();
}

// ==========================================
// 4. FILE HANDLING
// ==========================================
sealDropzone.addEventListener('click', () => fileInput.click());

sealDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    sealDropzone.classList.add('drag-over');
});
sealDropzone.addEventListener('dragleave', () => sealDropzone.classList.remove('drag-over'));
sealDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    sealDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});
filedRemove.addEventListener('click', (e) => { e.stopPropagation(); clearFile(); });

function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showToast('The Court only accepts PDF documents.', 'error'); return;
    }
    if (file.size > 50 * 1024 * 1024) {
        showToast('Document exceeds the 50 MB filing limit.', 'error'); return;
    }
    selectedFile = file;
    filedName.textContent = file.name;
    filedSize.textContent = formatBytes(file.size);
    filedDoc.classList.remove('hidden');
    sealDropzone.style.display = 'none';
    gavelBtn.disabled = false;
    showToast(`Document "${file.name}" has been filed.`, 'success');
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    filedDoc.classList.add('hidden');
    sealDropzone.style.display = '';
    gavelBtn.disabled = true;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ==========================================
// 5. AUDIT
// ==========================================
gavelBtn.addEventListener('click', commenceAudit);

async function commenceAudit() {
    if (!selectedFile) return;

    chamberIdle.classList.add('hidden');
    chamberResults.classList.add('hidden');
    chamberDelib.classList.remove('hidden');
    gavelBtn.disabled = true;
    courtBreakdown.classList.add('hidden');
    document.getElementById('post-audit-actions').classList.add('hidden');
    document.getElementById('quote-summary').classList.add('hidden');

    ['ds-1','ds-2','ds-3','ds-4','ds-5','ds-6'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.classList.remove('active', 'done'); }
    });
    document.getElementById('ds-1').classList.add('active');
    animateDeliberation();

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const resp = await fetch(`${API_BASE}/audit-document`, { method: 'POST', body: formData });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Court error: ${resp.status}`);
        }

        const data = await resp.json();
        auditData = data.results || [];
        lastAuditResponse = data;
        finishDeliberation();

        setTimeout(() => {
            renderJudgments(data);
            saveToHistory(data, selectedFile.name);
            const scCount = data.supreme_court_count || 0;
            const hcCount = data.high_court_count || 0;
            showToast(`Judgment delivered: ${data.total_citations_found || 0} citations reviewed (${scCount} SC, ${hcCount} HC).`, 'info');
        }, 700);

    } catch (error) {
        showToast(`Audit failed: ${error.message}`, 'error');
        chamberDelib.classList.add('hidden');
        chamberIdle.classList.remove('hidden');
    } finally {
        gavelBtn.disabled = false;
    }
}

function animateDeliberation() {
    const steps = ['ds-1', 'ds-2', 'ds-3', 'ds-4', 'ds-5', 'ds-6'];
    const texts = [
        ['READING DOCUMENT', 'Extracting text from filed PDF...'],
        ['IDENTIFYING AUTHORITIES', 'AI is finding all cited cases and attributed claims...'],
        ['CLASSIFYING COURTS', 'Separating High Court and Supreme Court citations...'],
        ['SEARCHING COURT RECORDS', 'Cross-referencing SC cases against the archive...'],
        ['RAG QUOTE VERIFICATION', 'Embedding quotes & searching source judgments...'],
        ['PRONOUNCING JUDGMENT', 'Running hallucination detection...']
    ];
    steps.forEach((id, i) => {
        setTimeout(() => {
            const el = document.getElementById(id);
            if (!el) return;
            if (i > 0) {
                const prevEl = document.getElementById(steps[i - 1]);
                if (prevEl) {
                    prevEl.classList.remove('active');
                    prevEl.classList.add('done');
                }
            }
            el.classList.add('active');
            document.getElementById('delib-title').textContent = texts[i][0];
            document.getElementById('delib-sub').textContent = texts[i][1];
        }, i * 1500);
    });
}

function finishDeliberation() {
    ['ds-1','ds-2','ds-3','ds-4','ds-5','ds-6'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove('active');
            el.classList.add('done');
        }
    });
    document.getElementById('delib-title').textContent = 'JUDGMENT READY';
    document.getElementById('delib-sub').textContent = 'The bench has concluded its review.';
}

// ==========================================
// 6. RENDER RESULTS
// ==========================================
function renderJudgments(data) {
    chamberDelib.classList.add('hidden');
    chamberResults.classList.remove('hidden');

    const results = data.results || [];
    const scCount = data.supreme_court_count || 0;
    const hcCount = data.high_court_count || 0;
    const total   = data.total_citations_found || results.length;

    let upheld = 0, fabricated = 0, skipped = 0, unverified = 0;
    let quoteVerified = 0, quoteContradicted = 0, quoteUnsupported = 0, quoteFabricated = 0;
    let quoteIssueCount = 0;
    
    judgmentRoll.innerHTML = '';

    results.forEach((item, i) => {
        const status = classifyStatus(item);
        if (status === 'verified')      upheld++;
        else if (status === 'hallucinated') fabricated++;
        else if (status === 'skipped')  skipped++;
        else                            unverified++;
        
        // Count quote verification stats
        const qv = item.quote_verification || {};
        const qStatus = (qv.status || '').toLowerCase();
        if (qStatus.includes('verified')) quoteVerified++;
        else if (qStatus.includes('contradicted')) { quoteContradicted++; quoteIssueCount++; }
        else if (qStatus.includes('unsupported')) { quoteUnsupported++; quoteIssueCount++; }
        else if (qStatus.includes('fabricated')) { quoteFabricated++; quoteIssueCount++; }
        
        const hasQuoteIssue = qStatus.includes('contradicted') || qStatus.includes('unsupported') || qStatus.includes('fabricated');
        const card = buildJudgmentCard(item, i, status, hasQuoteIssue);
        judgmentRoll.appendChild(card);
    });

    animateNum(vcTotal, total);
    animateNum(vcUpheld, upheld);
    animateNum(vcOverruled, fabricated);
    animateNum(vcSkipped, skipped);
    animateNum(vcUnheard, unverified);

    // Quote verification summary
    const quoteSummaryEl = document.getElementById('quote-summary');
    if (quoteVerified + quoteContradicted + quoteUnsupported + quoteFabricated > 0) {
        quoteSummaryEl.classList.remove('hidden');
        animateNum(vcQuoteVerified, quoteVerified);
        animateNum(vcQuoteContradicted, quoteContradicted);
        animateNum(vcQuoteUnsupported, quoteUnsupported);
        animateNum(vcQuoteFabricated, quoteFabricated);
    }

    jfAll.textContent        = total;
    jfUpheld.textContent     = upheld;
    jfFabricated.textContent = fabricated;
    jfSkipped.textContent    = skipped;
    jfUnverified.textContent = unverified;
    jfQuoteIssues.textContent = quoteIssueCount;

    courtBreakdown.classList.remove('hidden');
    cbScCount.textContent = scCount;
    cbHcCount.textContent = hcCount;
    const maxCount = Math.max(scCount, hcCount, 1);
    cbScFill.style.width = `${(scCount / maxCount) * 100}%`;
    cbHcFill.style.width = `${(hcCount / maxCount) * 100}%`;

    document.getElementById('post-audit-actions').classList.remove('hidden');
}

function classifyStatus(item) {
    const raw = (item.verification?.status || '').toLowerCase();
    const courtType = (item.court_type || '').toLowerCase();
    if (raw.includes('skipped') || (raw.includes('⚠️') && !raw.includes('hc-'))) return 'skipped';
    if (courtType === 'high court' && !raw.includes('verified') && !raw.includes('hallucination') && !raw.includes('hc-')) return 'skipped';
    if (raw.includes('verified') || raw.includes('🟢')) return 'verified';
    if (raw.includes('hallucination') || raw.includes('🔴')) return 'hallucinated';
    if (raw.includes('hc-')) {
        if (raw.includes('verified')) return 'verified';
        if (raw.includes('hallucination')) return 'hallucinated';
    }
    return 'no-match';
}

function classifyQuoteStatus(quoteVerification) {
    if (!quoteVerification) return null;
    const s = (quoteVerification.status || '').toLowerCase();
    if (s.includes('verified')) return 'quote-ok';
    if (s.includes('contradicted')) return 'quote-contradicted';
    if (s.includes('unsupported')) return 'quote-unsupported';
    if (s.includes('fabricated')) return 'quote-fabricated';
    if (s.includes('skipped')) return 'quote-skipped';
    if (s.includes('error')) return 'quote-error';
    return null;
}

function buildJudgmentCard(item, index, status, hasQuoteIssue) {
    const card = document.createElement('div');
    card.className = `j-card ${status}${hasQuoteIssue ? ' quote-issue' : ''}`;
    card.style.animationDelay = `${index * 0.08}s`;
    card.dataset.status = status;
    card.dataset.hasQuoteIssue = hasQuoteIssue ? 'true' : 'false';

    const v = item.verification || {};
    const qv = item.quote_verification || {};
    const verdictLabels = {
        'verified': 'UPHELD', 'hallucinated': 'FABRICATED',
        'skipped': 'HIGH COURT', 'no-match': 'UNVERIFIED'
    };
    const courtType   = item.court_type || 'Unknown';
    const matchedName = v.matched_name || '—';
    const reason      = v.reason || v.message || 'No observations by the Court.';
    const confidence  = typeof v.confidence === 'number' ? v.confidence : null;
    const courtIcon   = courtType.toLowerCase().includes('high')
        ? '<i class="fas fa-university"></i>'
        : '<i class="fas fa-landmark"></i>';
    
    const attributedClaim = item.attributed_claim || '';

    const confidenceBar = confidence !== null ? `
        <div class="confidence-bar-wrap">
            <span class="confidence-label">AI Confidence</span>
            <div class="confidence-track">
                <div class="confidence-fill ${confidence >= 80 ? 'high' : confidence >= 50 ? 'mid' : 'low'}" 
                     style="width:${confidence}%"></div>
            </div>
            <span class="confidence-pct">${confidence}%</span>
        </div>` : '';

    // Quote verification badge
    const qvStatus = classifyQuoteStatus(qv);
    const qvBadgeMap = {
        'quote-ok': '<span class="qv-badge qv-ok"><i class="fas fa-check-double"></i> Quote Verified</span>',
        'quote-contradicted': '<span class="qv-badge qv-bad"><i class="fas fa-exclamation-triangle"></i> Quote Contradicted</span>',
        'quote-unsupported': '<span class="qv-badge qv-warn"><i class="fas fa-question-circle"></i> Quote Unsupported</span>',
        'quote-fabricated': '<span class="qv-badge qv-bad"><i class="fas fa-ghost"></i> Quote Fabricated</span>',
        'quote-skipped': '<span class="qv-badge qv-skip"><i class="fas fa-forward"></i> Quote Check Skipped</span>',
        'quote-error': '<span class="qv-badge qv-skip"><i class="fas fa-bug"></i> Quote Check Error</span>'
    };
    const qvBadge = qvBadgeMap[qvStatus] || '';

    // Attributed claim preview
    const claimPreview = attributedClaim 
        ? `<div class="j-detail j-claim-preview">
                <i class="fas fa-quote-left"></i>
                <span class="j-label">CLAIM</span>
                <span class="j-value">"${esc(truncate(attributedClaim, 80))}"</span>
           </div>` 
        : '';

    card.innerHTML = `
        <div class="j-card-top">
            <div class="j-case-name">${esc(item.target_citation)}</div>
            <div class="j-badges">
                <span class="j-verdict-badge">${verdictLabels[status]}</span>
                ${qvBadge}
            </div>
        </div>
        ${confidenceBar}
        <div class="j-card-details">
            <div class="j-detail">
                ${courtIcon}
                <span class="j-label">COURT</span>
                <span class="j-value">${esc(courtType)}</span>
            </div>
            ${status === 'verified' ? `
                <div class="j-detail">
                    <i class="fas fa-gavel"></i>
                    <span class="j-label">MATCH</span>
                    <span class="j-value">${esc(matchedName)}</span>
                </div>
            ` : ''}
            ${claimPreview}
            <div class="j-detail">
                <i class="fas fa-feather-alt"></i>
                <span class="j-label">NOTE</span>
                <span class="j-value">${esc(truncate(reason, 90))}</span>
            </div>
        </div>
        <div class="j-card-foot">
            <div class="j-read-order">
                <i class="fas fa-scroll"></i> READ FULL ORDER
            </div>
            <span class="j-serial">MATTER #${String(index + 1).padStart(3, '0')}</span>
        </div>
    `;
    card.addEventListener('click', () => openOrder(item, status));
    return card;
}

// ==========================================
// 7. FILTER TABS
// ==========================================
document.querySelectorAll('.jf-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.jf-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const filter = tab.dataset.filter;
        document.querySelectorAll('.j-card').forEach(card => {
            if (filter === 'all') {
                card.style.display = '';
            } else if (filter === 'quote-issue') {
                card.style.display = card.dataset.hasQuoteIssue === 'true' ? '' : 'none';
            } else {
                card.style.display = card.dataset.status === filter ? '' : 'none';
            }
        });
    });
});

// ==========================================
// 8. ORDER MODAL
// ==========================================
function openOrder(item, status) {
    const v = item.verification || {};
    const qv = item.quote_verification || {};
    const courtType = item.court_type || 'Unknown';
    const confidence = typeof v.confidence === 'number' ? v.confidence : null;
    const attributedClaim = item.attributed_claim || '';

    const verdictText = {
        'verified': '✅ CITATION UPHELD — Exists in Supreme Court Records',
        'hallucinated': '❌ CITATION FABRICATED — Not Found in Any Record',
        'skipped': '⚠️ HIGH COURT CITATION — Bypassed Supreme Court Verification',
        'no-match': '⚠️ CITATION UNVERIFIED — No Matching Candidates Found'
    };

    let sectionIdx = 0;
    const nextSection = () => ['I','II','III','IV','V','VI','VII','VIII'][sectionIdx++];
    let html = '';

    html += `<div class="order-section"><div class="order-verdict-banner ${status}">${verdictText[status]}</div></div>`;

    if (confidence !== null) {
        html += `
        <div class="order-section">
            <div class="order-section-title">${nextSection()}. AI CONFIDENCE SCORE</div>
            <div class="confidence-bar-wrap" style="padding:0.5rem 0;">
                <div class="confidence-track" style="height:12px;">
                    <div class="confidence-fill ${confidence >= 80 ? 'high' : confidence >= 50 ? 'mid' : 'low'}" 
                         style="width:${confidence}%;height:100%;"></div>
                </div>
                <span class="confidence-pct" style="font-size:1.2rem;font-weight:700;color:var(--gold);">${confidence}%</span>
            </div>
        </div>`;
    }

    html += `
        <div class="order-section">
            <div class="order-section-title">${nextSection()}. CITATION AS SUBMITTED</div>
            <div class="order-field">
                <div class="of-label">CASE NAME / REFERENCE</div>
                <div class="of-value">${esc(item.target_citation)}</div>
            </div>
            <div class="order-field">
                <div class="of-label">COURT CLASSIFICATION</div>
                <div class="of-value">
                    ${courtType.toLowerCase().includes('high')
                        ? '<i class="fas fa-university" style="color:var(--amber);margin-right:6px;"></i>'
                        : '<i class="fas fa-landmark" style="color:var(--gold);margin-right:6px;"></i>'}
                    ${esc(courtType)}
                </div>
            </div>
        </div>`;

    // Attributed claim section
    if (attributedClaim) {
        html += `
            <div class="order-section">
                <div class="order-section-title">${nextSection()}. ATTRIBUTED CLAIM / QUOTE</div>
                <div class="order-field">
                    <div class="of-label">WHAT THE LAWYER CLAIMED THIS CASE STATES</div>
                    <div class="of-value order-quote-block">
                        <i class="fas fa-quote-left" style="color:var(--gold);opacity:0.5;margin-right:6px;"></i>
                        "${esc(attributedClaim)}"
                    </div>
                </div>
            </div>`;
    }

    if (status === 'verified') {
        html += `
            <div class="order-section">
                <div class="order-section-title">${nextSection()}. MATCHING RECORD</div>
                <div class="order-field">
                    <div class="of-label">CASE ON RECORD</div>
                    <div class="of-value">${esc(v.matched_name || '—')}</div>
                </div>
                <div class="order-field">
                    <div class="of-label">SOURCE FILE</div>
                    <div class="of-value" style="font-family:var(--font-mono);font-size:0.8rem;color:var(--gold);">
                        ${esc(v.file_to_open || '—')}
                    </div>
                </div>
            </div>`;
    }

    // QUOTE VERIFICATION SECTION (RAG)
    if (qv && qv.status) {
        const qvStatusClass = classifyQuoteStatus(qv);
        const qvStatusColors = {
            'quote-ok': '#4caf8a',
            'quote-contradicted': '#e87777',
            'quote-unsupported': '#e8b877',
            'quote-fabricated': '#e87777',
            'quote-skipped': '#aaa',
            'quote-error': '#aaa'
        };
        
        html += `
            <div class="order-section order-quote-section">
                <div class="order-section-title">${nextSection()}. QUOTE VERIFICATION (RAG ENGINE)</div>
                <div class="order-field">
                    <div class="of-label">VERIFICATION STATUS</div>
                    <div class="of-value" style="color:${qvStatusColors[qvStatusClass] || 'var(--text-primary)'}; font-weight:600;">
                        ${esc(qv.status)}
                    </div>
                </div>
                <div class="order-field">
                    <div class="of-label">REASONING</div>
                    <div class="of-value" style="font-style:italic;">
                        "${esc(qv.reason || qv.explanation || 'No reasoning provided.')}"
                    </div>
                </div>`;
        
        if (qv.found_paragraph) {
            html += `
                <div class="order-field">
                    <div class="of-label">MOST RELEVANT PARAGRAPH FROM SOURCE JUDGMENT</div>
                    <div class="of-value order-source-paragraph">
                        ${esc(qv.found_paragraph)}
                    </div>
                </div>`;
        }
        
        if (qv.closest_text_found) {
            html += `
                <div class="order-field">
                    <div class="of-label">CLOSEST TEXT FOUND (LOW SIMILARITY)</div>
                    <div class="of-value order-source-paragraph" style="border-color:rgba(232,119,119,0.3);">
                        ${esc(qv.closest_text_found)}
                    </div>
                </div>`;
        }
        
        html += `</div>`;
    }

    if (status === 'skipped') {
        html += `
            <div class="order-section">
                <div class="order-section-title">${nextSection()}. HIGH COURT DISPOSITION</div>
                <div class="order-field">
                    <div class="of-label">STATUS</div>
                    <div class="of-value" style="color:var(--amber);">
                        <i class="fas fa-exclamation-triangle" style="margin-right:6px;"></i>
                        This citation was identified as a High Court case by the AI classifier.
                    </div>
                </div>
                <div class="order-field">
                    <div class="of-label">RECOMMENDATION</div>
                    <div class="of-value" style="font-style:italic;">
                        Verify this citation against the relevant High Court database independently.
                    </div>
                </div>
            </div>`;
    }

    html += `
        <div class="order-section">
            <div class="order-section-title">${nextSection()}. OBSERVATIONS OF THE COURT</div>
            <div class="order-field">
                <div class="of-label">AI REASONING</div>
                <div class="of-value" style="font-style:italic;">
                    "${esc(v.reason || v.message || 'The Court has no further observations.')}"
                </div>
            </div>
        </div>
        <div class="order-section">
            <div class="order-section-title">${nextSection()}. VERIFICATION DATA</div>
            <div class="order-field">
                <div class="of-label">RAW JSON RESPONSE</div>
                <pre style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-secondary);white-space:pre-wrap;word-break:break-all;line-height:1.6;margin:0;">${esc(JSON.stringify({ 
                    court_type: courtType, 
                    attributed_claim: attributedClaim || undefined,
                    verification: v,
                    quote_verification: Object.keys(qv).length > 0 ? qv : undefined
                }, null, 2))}</pre>
            </div>
        </div>`;

    orderBody.innerHTML = html;
    orderOverlay.classList.remove('hidden');
}

orderClose.addEventListener('click', () => orderOverlay.classList.add('hidden'));
orderOverlay.addEventListener('click', (e) => { if (e.target === orderOverlay) orderOverlay.classList.add('hidden'); });
document.addEventListener('keydown', (e) => { 
    if (e.key === 'Escape') { 
        orderOverlay.classList.add('hidden'); 
        document.getElementById('summary-overlay').classList.add('hidden'); 
    } 
});

// ==========================================
// 9. MANUAL CITATION SEARCH
// ==========================================
async function runManualSearch() {
    const input = document.getElementById('manual-search-input');
    const btn = document.getElementById('search-btn');
    const area = document.getElementById('search-results-area');
    const query = input.value.trim();

    if (!query) { showToast('Please enter a citation to search.', 'warning'); return; }

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Verifying...';
    area.innerHTML = `<div class="search-loading"><i class="fas fa-gavel fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Cross-referencing with court archive...</p></div>`;

    try {
        const resp = await fetch(`${API_BASE}/verify-citation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ citation: query })
        });
        if (!resp.ok) throw new Error(`Error: ${resp.status}`);
        const data = await resp.json();
        renderSearchResult(area, data);
    } catch (err) {
        area.innerHTML = `<div class="search-error"><i class="fas fa-exclamation-triangle"></i><p>Search failed: ${esc(err.message)}</p></div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-gavel"></i> VERIFY';
    }
}

function fillSearch(text) {
    document.getElementById('manual-search-input').value = text;
    runManualSearch();
}

function renderSearchResult(area, data) {
    const v = data.verification || {};
    const status = classifyStatus(data);
    const confidence = typeof v.confidence === 'number' ? v.confidence : null;

    const statusColors = {
        'verified': '#4caf8a', 'hallucinated': '#e87777',
        'skipped': '#e8b877', 'no-match': '#aaa'
    };
    const statusLabels = {
        'verified': '🟢 VERIFIED', 'hallucinated': '🔴 FABRICATED',
        'skipped': '⚠️ HIGH COURT', 'no-match': '❓ UNVERIFIED'
    };

    area.innerHTML = `
        <div class="search-result-card ${status}">
            <div class="src-header">
                <div class="src-citation">${esc(data.target_citation)}</div>
                <span class="src-verdict" style="color:${statusColors[status]}">${statusLabels[status]}</span>
            </div>
            ${confidence !== null ? `
            <div class="confidence-bar-wrap" style="margin:0.75rem 0;">
                <span class="confidence-label">AI Confidence</span>
                <div class="confidence-track"><div class="confidence-fill ${confidence >= 80 ? 'high' : confidence >= 50 ? 'mid' : 'low'}" style="width:${confidence}%"></div></div>
                <span class="confidence-pct">${confidence}%</span>
            </div>` : ''}
            <div class="src-details">
                <div class="src-field"><span class="src-label">COURT</span><span>${esc(data.court_type)}</span></div>
                ${v.matched_name ? `<div class="src-field"><span class="src-label">MATCH</span><span>${esc(v.matched_name)}</span></div>` : ''}
                ${v.file_to_open ? `<div class="src-field"><span class="src-label">FILE</span><span style="font-family:var(--font-mono);font-size:0.8rem;color:var(--gold)">${esc(v.file_to_open)}</span></div>` : ''}
                <div class="src-field"><span class="src-label">REASON</span><span style="font-style:italic">${esc(v.reason || v.message || '—')}</span></div>
            </div>
        </div>`;
}

// ==========================================
// 10. BULK UPLOAD
// ==========================================
let bulkFiles = [];
const bulkFileInput = document.getElementById('bulk-file-input');
const bulkDropzone  = document.getElementById('bulk-dropzone');
const bulkAuditBtn  = document.getElementById('bulk-audit-btn');
const bulkFileList  = document.getElementById('bulk-file-list');

bulkDropzone.addEventListener('dragover', (e) => { e.preventDefault(); bulkDropzone.classList.add('drag-over'); });
bulkDropzone.addEventListener('dragleave', () => bulkDropzone.classList.remove('drag-over'));
bulkDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    bulkDropzone.classList.remove('drag-over');
    addBulkFiles([...e.dataTransfer.files]);
});
bulkFileInput.addEventListener('change', (e) => addBulkFiles([...e.target.files]));

function addBulkFiles(files) {
    const pdfs = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length === 0) { showToast('Please select PDF files only.', 'error'); return; }
    bulkFiles = [...bulkFiles, ...pdfs].slice(0, 10);
    renderBulkFileList();
    bulkAuditBtn.disabled = bulkFiles.length === 0;
}

function renderBulkFileList() {
    if (bulkFiles.length === 0) { bulkFileList.innerHTML = ''; return; }
    bulkFileList.innerHTML = bulkFiles.map((f, i) => `
        <div class="bulk-file-item">
            <i class="fas fa-file-pdf" style="color:var(--gold)"></i>
            <span class="bfi-name">${esc(f.name)}</span>
            <span class="bfi-size">${formatBytes(f.size)}</span>
            <button class="bfi-remove" onclick="removeBulkFile(${i})"><i class="fas fa-times"></i></button>
        </div>`).join('');
}

function removeBulkFile(idx) {
    bulkFiles.splice(idx, 1);
    renderBulkFileList();
    bulkAuditBtn.disabled = bulkFiles.length === 0;
}

async function runBulkAudit() {
    if (bulkFiles.length === 0) return;

    const progressEl = document.getElementById('bulk-progress');
    const progressFill = document.getElementById('bulk-progress-fill');
    const progressText = document.getElementById('bulk-progress-text');
    const resultsArea = document.getElementById('bulk-results-area');

    progressEl.classList.remove('hidden');
    bulkAuditBtn.disabled = true;
    resultsArea.innerHTML = '';

    const allResults = [];

    for (let i = 0; i < bulkFiles.length; i++) {
        const file = bulkFiles[i];
        const pct = Math.round(((i + 0.5) / bulkFiles.length) * 100);
        progressFill.style.width = `${pct}%`;
        progressText.textContent = `Processing ${i + 1}/${bulkFiles.length}: ${file.name}...`;

        try {
            const formData = new FormData();
            formData.append('file', file);
            const resp = await fetch(`${API_BASE}/audit-document`, { method: 'POST', body: formData });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            allResults.push({ ...data, filename: file.name, success: true });
        } catch (err) {
            allResults.push({ filename: file.name, success: false, error: err.message });
        }
    }

    progressFill.style.width = '100%';
    progressText.textContent = 'All documents processed!';

    renderBulkResults(allResults, resultsArea);
    bulkAuditBtn.disabled = false;

    setTimeout(() => progressEl.classList.add('hidden'), 2000);
}

function renderBulkResults(results, container) {
    const totalCitations = results.reduce((s, r) => s + (r.total_citations_found || 0), 0);
    const totalVerified  = results.reduce((s, r) => s + (r.results || []).filter(x => classifyStatus(x) === 'verified').length, 0);
    const totalFabricatedC = results.reduce((s, r) => s + (r.results || []).filter(x => classifyStatus(x) === 'hallucinated').length, 0);
    const totalQuoteIssues = results.reduce((s, r) => s + (r.results || []).filter(x => {
        const qs = (x.quote_verification?.status || '').toLowerCase();
        return qs.includes('contradicted') || qs.includes('unsupported') || qs.includes('fabricated');
    }).length, 0);

    container.innerHTML = `
        <div class="bulk-summary-header">
            <div class="bulk-stat"><div class="bs-num">${results.length}</div><div class="bs-label">Documents</div></div>
            <div class="bulk-stat"><div class="bs-num">${totalCitations}</div><div class="bs-label">Citations</div></div>
            <div class="bulk-stat upheld"><div class="bs-num">${totalVerified}</div><div class="bs-label">Verified</div></div>
            <div class="bulk-stat fabricated"><div class="bs-num">${totalFabricatedC}</div><div class="bs-label">Fabricated</div></div>
            <div class="bulk-stat quote-issues"><div class="bs-num">${totalQuoteIssues}</div><div class="bs-label">Quote Issues</div></div>
        </div>
        ${results.map(r => `
        <div class="bulk-doc-card ${r.success ? '' : 'error'}">
            <div class="bdc-header">
                <i class="fas fa-file-pdf" style="color:var(--gold)"></i>
                <span class="bdc-name">${esc(r.filename)}</span>
                ${r.success ? `<span class="bdc-badge">${r.total_citations_found || 0} citations</span>` : `<span class="bdc-badge error">ERROR</span>`}
            </div>
            ${r.success ? `
            <div class="bdc-stats">
                <span>✅ ${(r.results || []).filter(x => classifyStatus(x) === 'verified').length} verified</span>
                <span>❌ ${(r.results || []).filter(x => classifyStatus(x) === 'hallucinated').length} fabricated</span>
                <span>⚠️ ${(r.results || []).filter(x => classifyStatus(x) === 'skipped').length} HC</span>
                <span>📝 ${(r.results || []).filter(x => {
                    const qs = (x.quote_verification?.status || '').toLowerCase();
                    return qs.includes('contradicted') || qs.includes('fabricated');
                }).length} quote issues</span>
            </div>` : `<p style="color:#e87777;font-size:0.8rem;">${esc(r.error)}</p>`}
        </div>`).join('')}`;
}

// ==========================================
// 11. EXPORT FUNCTIONS
// ==========================================
function exportCSV() {
    if (!auditData.length) { showToast('No audit data to export.', 'warning'); return; }
    const rows = [['#', 'Citation', 'Court Type', 'Status', 'Confidence', 'Matched Name', 'Reason/Message', 'Attributed Claim', 'Quote Status', 'Quote Reason']];
    auditData.forEach((item, i) => {
        const v = item.verification || {};
        const qv = item.quote_verification || {};
        rows.push([
            i + 1,
            `"${(item.target_citation || '').replace(/"/g, '""')}"`,
            item.court_type || '',
            v.status || '',
            v.confidence ?? '',
            v.matched_name || '',
            `"${(v.reason || v.message || '').replace(/"/g, '""')}"`,
            `"${(item.attributed_claim || '').replace(/"/g, '""')}"`,
            qv.status || '',
            `"${(qv.reason || qv.explanation || '').replace(/"/g, '""')}"`
        ]);
    });
    const csv = rows.map(r => r.join(',')).join('\n');
    downloadFile(csv, 'citation_audit_report.csv', 'text/csv');
    showToast('CSV exported successfully.', 'success');
}

function exportPDF() {
    if (!auditData.length) { showToast('No audit data to export.', 'warning'); return; }
    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        doc.setFontSize(18);
        doc.text('Legal Citation Audit Report', 20, 20);
        doc.setFontSize(10);
        doc.text(`Generated: ${new Date().toLocaleString('en-IN')}`, 20, 28);
        doc.text(`Session: ${sessionIdEl.textContent}`, 20, 34);
        
        let y = 45;
        
        // Summary
        const verified = auditData.filter(x => classifyStatus(x) === 'verified').length;
        const fabricated = auditData.filter(x => classifyStatus(x) === 'hallucinated').length;
        const quoteIssues = auditData.filter(x => {
            const qs = (x.quote_verification?.status || '').toLowerCase();
            return qs.includes('contradicted') || qs.includes('fabricated');
        }).length;
        
        doc.setFontSize(12);
        doc.text('Summary', 20, y); y += 8;
        doc.setFontSize(9);
        doc.text(`Total Citations: ${auditData.length}`, 25, y); y += 5;
        doc.text(`Verified: ${verified}  |  Fabricated: ${fabricated}  |  Quote Issues: ${quoteIssues}`, 25, y); y += 10;
        
        doc.setFontSize(12);
        doc.text('Detailed Results', 20, y); y += 8;
        
        auditData.forEach((item, i) => {
            if (y > 270) { doc.addPage(); y = 20; }
            const v = item.verification || {};
            const qv = item.quote_verification || {};
            doc.setFontSize(9);
            doc.setFont(undefined, 'bold');
            doc.text(`${i + 1}. ${truncate(item.target_citation || '', 70)}`, 20, y); y += 5;
            doc.setFont(undefined, 'normal');
            doc.text(`   Status: ${v.status || 'Unknown'}  |  Court: ${item.court_type || 'Unknown'}`, 20, y); y += 5;
            if (v.matched_name) {
                doc.text(`   Match: ${v.matched_name}`, 20, y); y += 5;
            }
            if (item.attributed_claim) {
                const claimText = truncate(item.attributed_claim, 80);
                doc.text(`   Claim: "${claimText}"`, 20, y); y += 5;
            }
            if (qv.status) {
                doc.text(`   Quote: ${qv.status}`, 20, y); y += 5;
            }
            y += 3;
        });
        
        doc.save('citation_audit_report.pdf');
        showToast('PDF exported successfully.', 'success');
    } catch (e) {
        showToast('PDF export failed: ' + e.message, 'error');
    }
}

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

// ==========================================
// 12. AI SUMMARY
// ==========================================
async function generateSummary() {
    if (!auditData.length) { showToast('No audit data available.', 'warning'); return; }
    
    const summaryOverlay = document.getElementById('summary-overlay');
    const summaryBody = document.getElementById('summary-body');
    summaryOverlay.classList.remove('hidden');
    summaryBody.innerHTML = `<div class="summary-loading"><i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Generating professional summary with quote verification analysis...</p></div>`;

    const scCount = lastAuditResponse?.supreme_court_count || 0;
    const hcCount = lastAuditResponse?.high_court_count || 0;

    try {
        const resp = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: auditData,
                total: auditData.length,
                sc_count: scCount,
                hc_count: hcCount
            })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        
        const riskColors = { 'Low': '#4caf8a', 'Medium': '#e8b877', 'High': '#e87777' };
        
        summaryBody.innerHTML = `
            <div class="summary-content">
                <div class="summary-risk-badge" style="background:${riskColors[data.risk_level] || '#aaa'}20; border:1px solid ${riskColors[data.risk_level] || '#aaa'}; color:${riskColors[data.risk_level] || '#aaa'}; padding:0.5rem 1rem; border-radius:6px; text-align:center; font-weight:700; margin-bottom:1rem;">
                    RISK LEVEL: ${data.risk_level || 'Unknown'}
                </div>
                <div class="summary-stats" style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;margin-bottom:1rem;">
                    <div style="text-align:center;padding:0.5rem;background:rgba(76,175,138,0.1);border-radius:6px;">
                        <div style="font-size:1.2rem;font-weight:700;color:#4caf8a;">${data.stats?.verified || 0}</div>
                        <div style="font-size:0.7rem;color:var(--text-secondary);">Verified</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(232,119,119,0.1);border-radius:6px;">
                        <div style="font-size:1.2rem;font-weight:700;color:#e87777;">${data.stats?.fabricated || 0}</div>
                        <div style="font-size:0.7rem;color:var(--text-secondary);">Fabricated</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(232,184,119,0.1);border-radius:6px;">
                        <div style="font-size:1.2rem;font-weight:700;color:#e8b877;">${data.stats?.skipped || 0}</div>
                        <div style="font-size:0.7rem;color:var(--text-secondary);">HC Skipped</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(170,170,170,0.1);border-radius:6px;">
                        <div style="font-size:1.2rem;font-weight:700;color:#aaa;">${data.stats?.unverified || 0}</div>
                        <div style="font-size:0.7rem;color:var(--text-secondary);">Unverified</div>
                    </div>
                </div>
                <div class="summary-text" style="line-height:1.8;color:var(--text-primary);white-space:pre-wrap;">${esc(data.summary)}</div>
            </div>`;
        
        lastSummaryText = data.summary;
    } catch (err) {
        summaryBody.innerHTML = `<div class="search-error"><i class="fas fa-exclamation-triangle"></i><p>Summary generation failed: ${esc(err.message)}</p></div>`;
    }
}

let lastSummaryText = '';

function exportSummaryPDF() {
    if (!lastSummaryText) { showToast('No summary to export.', 'warning'); return; }
    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        doc.setFontSize(16);
        doc.text('AI Audit Summary Report', 20, 20);
        doc.setFontSize(10);
        doc.text(`Generated: ${new Date().toLocaleString('en-IN')}`, 20, 28);
        doc.setFontSize(10);
        const lines = doc.splitTextToSize(lastSummaryText, 170);
        doc.text(lines, 20, 40);
        doc.save('ai_audit_summary.pdf');
        showToast('Summary PDF exported.', 'success');
    } catch (e) {
        showToast('Export failed: ' + e.message, 'error');
    }
}

// ==========================================
// 13. HISTORY
// ==========================================
function saveToHistory(data, filename) {
    const history = JSON.parse(localStorage.getItem('auditHistory') || '[]');
    
    const quoteIssues = (data.results || []).filter(x => {
        const qs = (x.quote_verification?.status || '').toLowerCase();
        return qs.includes('contradicted') || qs.includes('fabricated');
    }).length;
    
    history.unshift({
        id: Date.now(),
        filename,
        date: new Date().toISOString(),
        total: data.total_citations_found || 0,
        sc: data.supreme_court_count || 0,
        hc: data.high_court_count || 0,
        verified: (data.results || []).filter(x => classifyStatus(x) === 'verified').length,
        fabricated: (data.results || []).filter(x => classifyStatus(x) === 'hallucinated').length,
        quoteIssues: quoteIssues,
        results: data.results
    });
    
    // Keep only last 50
    if (history.length > 50) history.length = 50;
    localStorage.setItem('auditHistory', JSON.stringify(history));
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem('auditHistory') || '[]');
    const list = document.getElementById('history-list');
    const count = document.getElementById('history-count');
    count.textContent = `${history.length} records`;

    if (history.length === 0) {
        list.innerHTML = `<div class="history-empty"><i class="fas fa-history" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i><p>No audit history yet.</p></div>`;
        return;
    }

    list.innerHTML = history.map(h => `
        <div class="history-item" onclick='loadHistoryItem(${h.id})'>
            <div class="hi-top">
                <div class="hi-file"><i class="fas fa-file-pdf" style="color:var(--gold);margin-right:6px;"></i>${esc(h.filename)}</div>
                <div class="hi-date">${new Date(h.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })}</div>
            </div>
            <div class="hi-stats">
                <span>📜 ${h.total} citations</span>
                <span>✅ ${h.verified} verified</span>
                <span>❌ ${h.fabricated} fabricated</span>
                <span>⚖️ ${h.sc} SC / ${h.hc} HC</span>
                ${h.quoteIssues ? `<span>📝 ${h.quoteIssues} quote issues</span>` : ''}
            </div>
        </div>`).join('');
}

function loadHistoryItem(id) {
    const history = JSON.parse(localStorage.getItem('auditHistory') || '[]');
    const item = history.find(h => h.id === id);
    if (!item) return;
    
    auditData = item.results || [];
    lastAuditResponse = {
        results: item.results,
        total_citations_found: item.total,
        supreme_court_count: item.sc,
        high_court_count: item.hc
    };
    
    switchTab('audit');
    renderJudgments(lastAuditResponse);
    showToast(`Loaded history: ${item.filename}`, 'info');
}

function clearHistory() {
    if (confirm('Clear all audit history?')) {
        localStorage.removeItem('auditHistory');
        renderHistory();
        showToast('History cleared.', 'info');
    }
}

// ==========================================
// 14. CHATBOT
// ==========================================
let chatHistory = [];
let useAuditContext = false;

function toggleChat() {
    const pane = document.getElementById('chat-pane');
    pane.classList.toggle('hidden');
    document.getElementById('chat-notification').style.display = 'none';
}

function toggleAuditContext() {
    useAuditContext = !useAuditContext;
    const bar = document.getElementById('chat-context-bar');
    const btn = document.getElementById('ctx-btn');
    if (useAuditContext) {
        bar.classList.remove('hidden');
        btn.classList.add('active');
        if (!lastAuditResponse) {
            showToast('No audit data yet. Run an audit first!', 'warning');
        }
    } else {
        bar.classList.add('hidden');
        btn.classList.remove('active');
    }
}

function clearChat() {
    chatHistory = [];
    const messages = document.getElementById('chat-messages');
    messages.innerHTML = `
        <div class="chat-msg assistant">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble">
                <p>Chat cleared. How can I help you?</p>
            </div>
        </div>`;
    document.getElementById('chat-suggestions').style.display = '';
}

function sendSuggestion(text) {
    document.getElementById('chat-input').value = text;
    sendChatMessage();
}

function handleChatKey(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

function autoResizeChat(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const messages = document.getElementById('chat-messages');
    const msg = input.value.trim();
    if (!msg) return;

    // Hide suggestions
    document.getElementById('chat-suggestions').style.display = 'none';

    // Add user message
    messages.innerHTML += `
        <div class="chat-msg user">
            <div class="msg-bubble">${esc(msg)}</div>
        </div>`;
    chatHistory.push({ role: 'user', content: msg });
    input.value = '';
    input.style.height = 'auto';

    // Scroll down
    messages.scrollTop = messages.scrollHeight;

    // Show typing indicator
    const typingId = 'typing-' + Date.now();
    messages.innerHTML += `
        <div class="chat-msg assistant" id="${typingId}">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    messages.scrollTop = messages.scrollHeight;

    // Build audit context string
    let auditCtx = null;
    if (useAuditContext && lastAuditResponse) {
        const results = lastAuditResponse.results || [];
        const verified = results.filter(x => classifyStatus(x) === 'verified').length;
        const fabricated = results.filter(x => classifyStatus(x) === 'hallucinated').length;
        const quoteIssues = results.filter(x => {
            const qs = (x.quote_verification?.status || '').toLowerCase();
            return qs.includes('contradicted') || qs.includes('fabricated');
        }).length;
        
        const fabricatedNames = results.filter(x => classifyStatus(x) === 'hallucinated').map(x => x.target_citation).slice(0, 5);
        const quoteIssueNames = results.filter(x => {
            const qs = (x.quote_verification?.status || '').toLowerCase();
            return qs.includes('contradicted') || qs.includes('fabricated');
        }).map(x => `${x.target_citation}: ${x.quote_verification?.status}`).slice(0, 5);
        
        auditCtx = `Last audit: ${results.length} citations total. ${verified} verified, ${fabricated} fabricated, ${quoteIssues} quote issues. ` +
            (fabricatedNames.length ? `Fabricated: ${fabricatedNames.join(', ')}. ` : '') +
            (quoteIssueNames.length ? `Quote issues: ${quoteIssueNames.join('; ')}. ` : '');
    }

    try {
        const resp = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                history: chatHistory.slice(-10),
                audit_context: auditCtx
            })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        
        // Remove typing indicator
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();

        // Add assistant response
        const reply = data.reply || 'I apologize, I could not generate a response.';
        messages.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-avatar">⚖️</div>
                <div class="msg-bubble">${formatChatResponse(reply)}</div>
            </div>`;
        chatHistory.push({ role: 'assistant', content: reply });
        messages.scrollTop = messages.scrollHeight;
    } catch (err) {
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();
        messages.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-avatar">⚖️</div>
                <div class="msg-bubble" style="color:#e87777;">Error: ${esc(err.message)}</div>
            </div>`;
        messages.scrollTop = messages.scrollHeight;
    }
}

function formatChatResponse(text) {
    // Convert markdown-like formatting
    let html = esc(text);
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Bullet points
    html = html.replace(/^- (.*)/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
}

// ==========================================
// 15. UTILITY FUNCTIONS
// ==========================================
function esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function truncate(str, maxLen) {
    if (!str) return '';
    return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}

function animateNum(el, target) {
    if (!el) return;
    const duration = 600;
    const start = parseInt(el.textContent) || 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        el.textContent = Math.round(start + (target - start) * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icons = { success: 'fa-check-circle', error: 'fa-times-circle', warning: 'fa-exclamation-triangle', info: 'fa-info-circle' };
    toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${esc(message)}</span>`;
    
    toastContainer.appendChild(toast);
    
    // Trigger animation
    requestAnimationFrame(() => toast.classList.add('show'));
    
    setTimeout(() => {
        toast.classList.remove('show');
        toast.classList.add('hide');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==========================================
// 16. INIT
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    runOathSequence();
});
```

---

### File: `frontend/css/styles.css`

```css
/* ==========================================
   HOME PAGE & BAIL RECKONER STYLES
   ========================================== */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --bg-card-hover: #1c1c28;
    --bg-input: #0f1118;
    --gold: #c9a84c;
    --gold-light: #e0c878;
    --gold-dark: #9a7b30;
    --gold-glow: rgba(201, 168, 76, 0.15);
    --green: #4caf8a;
    --green-bg: rgba(76, 175, 138, 0.1);
    --green-border: rgba(76, 175, 138, 0.3);
    --red: #e87777;
    --red-bg: rgba(232, 119, 119, 0.1);
    --red-border: rgba(232, 119, 119, 0.3);
    --yellow: #e8b877;
    --yellow-bg: rgba(232, 184, 119, 0.1);
    --yellow-border: rgba(232, 184, 119, 0.3);
    --blue: #77b8e8;
    --blue-bg: rgba(119, 184, 232, 0.1);
    --blue-border: rgba(119, 184, 232, 0.3);
    --text-primary: #e8e4dd;
    --text-secondary: #8a8578;
    --text-muted: #5a564f;
    --border-color: rgba(201, 168, 76, 0.12);
    --border-highlight: rgba(201, 168, 76, 0.25);
    --font-display: 'Playfair Display', serif;
    --font-body: 'Inter', sans-serif;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 20px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 8px 40px rgba(0, 0, 0, 0.5);
    --shadow-glow: 0 0 40px rgba(201, 168, 76, 0.1);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
    font-family: var(--font-body);
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
    overflow-x: hidden;
}
::selection { background: var(--gold); color: var(--bg-primary); }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-highlight); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* Background Animation */
.bg-animation { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; overflow: hidden; }
.grid-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(201,168,76,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(201,168,76,0.03) 1px, transparent 1px); background-size: 60px 60px; }
.floating-paragraphs { position: absolute; width: 100%; height: 100%; }
.para-symbol { position: absolute; font-size: 3rem; color: rgba(201,168,76,0.04); animation: float-symbol 20s infinite ease-in-out; }
.para-symbol:nth-child(1) { top: 10%; left: 5%; animation-delay: 0s; }
.para-symbol:nth-child(2) { top: 20%; left: 85%; animation-delay: 3s; }
.para-symbol:nth-child(3) { top: 50%; left: 15%; animation-delay: 6s; }
.para-symbol:nth-child(4) { top: 70%; left: 75%; animation-delay: 9s; }
.para-symbol:nth-child(5) { top: 30%; left: 45%; animation-delay: 2s; }
.para-symbol:nth-child(6) { top: 80%; left: 35%; animation-delay: 5s; }
.para-symbol:nth-child(7) { top: 15%; left: 65%; animation-delay: 8s; }
.para-symbol:nth-child(8) { top: 60%; left: 90%; animation-delay: 11s; }
@keyframes float-symbol { 0%, 100% { transform: translateY(0) rotate(0deg); opacity: 0.04; } 50% { transform: translateY(-30px) rotate(10deg); opacity: 0.08; } }

/* Navbar */
.navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 1000; background: rgba(10,10,15,0.85); backdrop-filter: blur(20px); border-bottom: 1px solid var(--border-color); padding: 0 2rem; height: 70px; }
.nav-container { max-width: 1400px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; height: 100%; }
.nav-logo { display: flex; align-items: center; gap: 12px; text-decoration: none; color: var(--text-primary); }
.logo-icon { width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, var(--gold), var(--gold-dark)); border-radius: 10px; font-size: 1.1rem; color: var(--bg-primary); box-shadow: 0 0 20px rgba(201,168,76,0.3); }
.logo-text { display: flex; flex-direction: column; }
.logo-main { font-family: var(--font-display); font-size: 1.3rem; font-weight: 700; letter-spacing: 1px; color: var(--gold); }
.logo-sub { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 2px; color: var(--text-muted); }
.nav-links { display: flex; gap: 8px; }
.nav-link { text-decoration: none; color: var(--text-secondary); padding: 8px 20px; border-radius: var(--radius-sm); font-size: 0.9rem; font-weight: 500; transition: var(--transition); position: relative; }
.nav-link:hover { color: var(--text-primary); background: rgba(255,255,255,0.05); }
.nav-link.active { color: var(--gold); background: var(--gold-glow); }
.nav-status { display: flex; align-items: center; gap: 8px; padding: 6px 16px; background: rgba(255,255,255,0.03); border-radius: 20px; border: 1px solid var(--border-color); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); transition: var(--transition); }
.status-dot.online { background: var(--green); box-shadow: 0 0 10px rgba(76,175,138,0.5); animation: pulse-dot 2s infinite; }
.status-dot.offline { background: var(--red); }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.status-text { font-size: 0.75rem; color: var(--text-muted); font-weight: 500; }

/* Hero */
.hero { position: relative; z-index: 1; min-height: 80vh; display: flex; align-items: center; justify-content: center; padding: 120px 2rem 60px; text-align: center; }
.hero-content { max-width: 800px; }
.hero-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 20px; background: var(--gold-glow); border: 1px solid var(--border-highlight); border-radius: 50px; font-size: 0.8rem; font-weight: 600; color: var(--gold); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 2rem; animation: fadeInUp 0.6s ease; }
.hero-title { font-family: var(--font-display); font-size: 4rem; font-weight: 800; line-height: 1.1; margin-bottom: 1.5rem; animation: fadeInUp 0.6s ease 0.1s both; }
.title-line { display: block; }
.title-accent { background: linear-gradient(135deg, var(--gold), var(--gold-light), var(--gold)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-description { font-size: 1.15rem; color: var(--text-secondary); line-height: 1.7; max-width: 600px; margin: 0 auto 3rem; animation: fadeInUp 0.6s ease 0.2s both; }
.hero-stats { display: flex; align-items: center; justify-content: center; gap: 3rem; animation: fadeInUp 0.6s ease 0.3s both; }
.stat-item { text-align: center; }
.stat-number { display: block; font-family: var(--font-display); font-size: 2.2rem; font-weight: 700; color: var(--gold); }
.stat-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
.stat-divider { width: 1px; height: 40px; background: var(--border-color); }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }

/* Service Cards */
.services { position: relative; z-index: 1; padding: 0 2rem 80px; }
.services-container { max-width: 1200px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
.service-card { position: relative; display: flex; flex-direction: column; padding: 2.5rem; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-xl); text-decoration: none; color: var(--text-primary); overflow: hidden; transition: all 0.5s cubic-bezier(0.4,0,0.2,1); cursor: pointer; }
.service-card:hover { transform: translateY(-8px); border-color: var(--border-highlight); box-shadow: var(--shadow-glow), var(--shadow-lg); }
.card-glow { position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(201,168,76,0.05) 0%, transparent 60%); opacity: 0; transition: all 0.5s; pointer-events: none; }
.service-card:hover .card-glow { opacity: 1; }
.card-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 1.5rem; }
.card-icon { position: relative; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: var(--gold); }
.icon-ring { position: absolute; inset: 0; border: 2px solid var(--gold); border-radius: 16px; opacity: 0.3; transition: var(--transition); }
.service-card:hover .icon-ring { opacity: 0.6; transform: scale(1.1) rotate(5deg); }
.card-badge { padding: 4px 12px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; color: var(--gold); background: var(--gold-glow); border: 1px solid var(--border-highlight); border-radius: 50px; }
.card-title { font-family: var(--font-display); font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem; }
.card-description { font-size: 0.95rem; color: var(--text-secondary); line-height: 1.7; margin-bottom: 1.5rem; }
.card-features { display: flex; flex-direction: column; gap: 10px; margin-bottom: 2rem; flex-grow: 1; }
.feature { display: flex; align-items: center; gap: 10px; font-size: 0.88rem; color: var(--text-secondary); }
.feature i { color: var(--gold); font-size: 0.75rem; }
.card-action { display: flex; align-items: center; gap: 10px; font-size: 0.95rem; font-weight: 600; color: var(--gold); padding-top: 1.5rem; border-top: 1px solid var(--border-color); transition: var(--transition); }
.card-action i { transition: var(--transition); }
.service-card:hover .card-action i { transform: translateX(6px); }

/* How It Works */
.how-it-works { position: relative; z-index: 1; padding: 80px 2rem; background: rgba(255,255,255,0.01); border-top: 1px solid var(--border-color); }
.section-container { max-width: 1200px; margin: 0 auto; }
.section-title { font-family: var(--font-display); font-size: 2.2rem; font-weight: 700; text-align: center; margin-bottom: 3rem; }
.title-decoration { color: var(--gold); margin-right: 10px; }
.steps-grid { display: flex; align-items: flex-start; justify-content: center; }
.step-card { flex: 1; max-width: 240px; text-align: center; padding: 2rem 1.5rem; position: relative; }
.step-number { font-family: var(--font-display); font-size: 3rem; font-weight: 900; color: var(--gold); opacity: 0.15; position: absolute; top: 10px; right: 15px; }
.step-icon { width: 60px; height: 60px; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; background: var(--gold-glow); border: 1px solid var(--border-highlight); border-radius: 50%; font-size: 1.3rem; color: var(--gold); }
.step-card h3 { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; }
.step-card p { font-size: 0.85rem; color: var(--text-muted); line-height: 1.6; }
.step-connector { display: flex; align-items: center; padding-top: 2.5rem; color: var(--gold); opacity: 0.3; font-size: 0.8rem; }

/* Footer */
.footer { position: relative; z-index: 1; padding: 2rem; border-top: 1px solid var(--border-color); background: rgba(0,0,0,0.3); }
.footer-container { max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; }
.footer-brand { display: flex; align-items: center; gap: 10px; color: var(--text-muted); font-size: 0.85rem; }
.footer-brand i { color: var(--gold); }
.footer-center { font-family: 'Cormorant Garamond', serif; font-size: 0.8rem; color: var(--text-muted); font-style: italic; }
.footer-disclaimer p { font-size: 0.75rem; color: var(--text-muted); }

/* ==========================================
   BAIL RECKONER STYLES
   ========================================== */
.bail-main { padding-top: 70px; min-height: 100vh; }
.bail-container { display: grid; grid-template-columns: 500px 1fr; min-height: calc(100vh - 70px); }
.bail-form-panel { background: var(--bg-secondary); border-right: 1px solid var(--border-color); padding: 2rem; overflow-y: auto; max-height: calc(100vh - 70px); }
.bail-results-panel { padding: 2rem; overflow-y: auto; max-height: calc(100vh - 70px); display: flex; flex-direction: column; align-items: center; justify-content: center; }
.form-header { text-align: center; margin-bottom: 2rem; padding-bottom: 2rem; border-bottom: 1px solid var(--border-color); }
.form-icon { width: 70px; height: 70px; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; background: var(--gold-glow); border: 2px solid var(--border-highlight); border-radius: 50%; font-size: 1.8rem; color: var(--gold); }
.form-header h1 { font-family: var(--font-display); font-size: 1.6rem; font-weight: 700; margin-bottom: 0.5rem; }
.form-header p { font-size: 0.85rem; color: var(--text-muted); }
.bail-form { display: flex; flex-direction: column; gap: 1.5rem; }
.form-group { display: flex; flex-direction: column; gap: 8px; }
.form-label { font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.form-label i { color: var(--gold); font-size: 0.8rem; }
.form-select, .form-input { padding: 12px 16px; background: var(--bg-input); border: 1px solid var(--border-color); border-radius: var(--radius-sm); color: var(--text-primary); font-size: 0.9rem; font-family: var(--font-body); transition: var(--transition); width: 100%; appearance: none; }
.form-select { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%238a8578' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 16px center; padding-right: 40px; }
.form-select:focus, .form-input:focus { outline: none; border-color: var(--gold); box-shadow: 0 0 0 3px var(--gold-glow); }
.form-select option { background: var(--bg-secondary); color: var(--text-primary); }
.input-hint { font-size: 0.75rem; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }
.input-hint i { color: var(--blue); }

/* Toggle switches */
.toggle-group { display: flex; flex-direction: column; gap: 12px; }
.toggle-item { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; background: var(--bg-input); border: 1px solid var(--border-color); border-radius: var(--radius-sm); transition: var(--transition); }
.toggle-item:hover { border-color: rgba(255,255,255,0.1); }
.toggle-info { display: flex; flex-direction: column; gap: 2px; }
.toggle-label { font-size: 0.88rem; font-weight: 500; }
.toggle-description { font-size: 0.73rem; color: var(--text-muted); }
.toggle-switch { position: relative; display: inline-block; width: 48px; height: 26px; flex-shrink: 0; }
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.toggle-slider { position: absolute; cursor: pointer; inset: 0; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 26px; transition: var(--transition); }
.toggle-slider::before { content: ''; position: absolute; width: 20px; height: 20px; left: 3px; bottom: 2px; background: var(--text-muted); border-radius: 50%; transition: var(--transition); }
.toggle-switch input:checked + .toggle-slider { background: var(--gold-glow); border-color: var(--gold); }
.toggle-switch input:checked + .toggle-slider::before { transform: translateX(22px); background: var(--gold); }
.btn-bail-submit { padding: 14px; font-size: 1rem; margin-top: 0.5rem; background: linear-gradient(135deg, var(--gold), var(--gold-dark)); border: none; border-radius: var(--radius-sm); color: var(--bg-primary); font-weight: 700; cursor: pointer; transition: var(--transition); display: flex; align-items: center; justify-content: center; gap: 8px; font-family: var(--font-body); }
.btn-bail-submit:hover { transform: translateY(-2px); box-shadow: 0 6px 25px rgba(201,168,76,0.4); }

/* Bail Results */
.bail-empty, .bail-loading, .bail-no-data { text-align: center; padding: 3rem 2rem; max-width: 400px; }
.gavel-animation i { font-size: 4rem; color: var(--gold); opacity: 0.2; animation: gavel-swing 3s ease-in-out infinite; }
@keyframes gavel-swing { 0%, 100% { transform: rotate(0deg); } 25% { transform: rotate(-15deg); } 75% { transform: rotate(15deg); } }
.bail-empty h3, .bail-loading h3, .bail-no-data h3 { font-family: var(--font-display); font-size: 1.3rem; margin-bottom: 0.5rem; color: var(--text-secondary); }
.bail-empty p, .bail-loading p, .bail-no-data p { font-size: 0.88rem; color: var(--text-muted); }
.bail-disclaimer { margin-top: 1.5rem; padding: 12px 16px; background: var(--yellow-bg); border: 1px solid var(--yellow-border); border-radius: var(--radius-sm); font-size: 0.78rem; color: var(--yellow); display: flex; align-items: flex-start; gap: 8px; text-align: left; }
.bail-results { width: 100%; max-width: 600px; animation: fadeInUp 0.5s ease; }
.result-section-title { font-family: var(--font-display); font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; }

/* Gauge */
.probability-section { margin-bottom: 2rem; text-align: center; }
.gauge-container { display: flex; justify-content: center; }
.gauge { position: relative; width: 220px; height: 140px; }
.gauge-svg { width: 100%; height: auto; }
.gauge-value { position: absolute; bottom: 15px; left: 50%; transform: translateX(-50%); font-family: var(--font-display); font-size: 2.2rem; font-weight: 800; color: var(--gold); }
.gauge-label { position: absolute; bottom: -5px; left: 50%; transform: translateX(-50%); font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; }
.metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem; }
.metric-card { padding: 1.5rem; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-md); text-align: center; }
.metric-icon { color: var(--gold); font-size: 1.2rem; margin-bottom: 0.5rem; }
.metric-value { font-family: var(--font-display); font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
.metric-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); }
.bond-section { margin-bottom: 2rem; }
.bond-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.bond-card { padding: 1.2rem; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-sm); display: flex; flex-direction: column; align-items: center; gap: 8px; text-align: center; transition: var(--transition); }
.bond-card i { font-size: 1.3rem; color: var(--text-muted); }
.bond-type { font-size: 0.82rem; font-weight: 600; color: var(--text-secondary); }
.bond-status { font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 50px; }
.bond-card.likely { border-color: var(--green-border); }
.bond-card.likely i { color: var(--green); }
.bond-card.likely .bond-status { background: var(--green-bg); color: var(--green); }
.bond-card.unlikely .bond-status { background: rgba(255,255,255,0.05); color: var(--text-muted); }
.strategy-note { background: var(--bg-card); border: 1px solid var(--border-highlight); border-radius: var(--radius-md); overflow: hidden; margin-bottom: 1rem; }
.strategy-header { display: flex; align-items: center; gap: 10px; padding: 12px 16px; background: var(--gold-glow); border-bottom: 1px solid var(--border-highlight); font-size: 0.85rem; font-weight: 600; color: var(--gold); }
.strategy-text { padding: 14px 16px; font-size: 0.88rem; color: var(--text-secondary); line-height: 1.6; }
.statutory-warning { display: flex; align-items: flex-start; gap: 12px; padding: 16px; background: var(--yellow-bg); border: 1px solid var(--yellow-border); border-radius: var(--radius-md); animation: fadeInUp 0.4s ease; }
.statutory-warning i { color: var(--yellow); font-size: 1.2rem; flex-shrink: 0; margin-top: 2px; }
.statutory-warning p { font-size: 0.88rem; color: var(--yellow); line-height: 1.6; font-weight: 500; }
.no-data-icon { font-size: 3rem; color: var(--text-muted); opacity: 0.3; margin-bottom: 1rem; }
.no-data-suggestion { margin-top: 0.5rem; font-size: 0.82rem; color: var(--blue); }
.loading-scale { width: 80px; height: 60px; margin: 0 auto; position: relative; }
.scale-beam { width: 80px; height: 4px; background: var(--gold); position: absolute; top: 10px; border-radius: 2px; animation: beam-tilt 2s ease-in-out infinite; transform-origin: center; }
@keyframes beam-tilt { 0%, 100% { transform: rotate(0deg); } 25% { transform: rotate(5deg); } 75% { transform: rotate(-5deg); } }

/* Responsive */
@media (max-width: 1100px) {
    .services-container { grid-template-columns: 1fr; max-width: 600px; }
    .bail-container { grid-template-columns: 1fr; }
    .bail-form-panel { max-height: none; border-right: none; border-bottom: 1px solid var(--border-color); }
    .bail-results-panel { max-height: none; min-height: 50vh; }
}
@media (max-width: 768px) {
    .hero-title { font-size: 2.5rem; }
    .hero-stats { flex-direction: column; gap: 1.5rem; }
    .stat-divider { width: 40px; height: 1px; }
    .steps-grid { flex-direction: column; align-items: center; }
    .step-connector { transform: rotate(90deg); padding: 0.5rem 0; }
    .navbar { padding: 0 1rem; }
    .nav-links { display: none; }
    .footer-container { flex-direction: column; text-align: center; }
}
```

---

### File: `frontend/static/app.js`

```js
// ==========================================
// ⚖️ CITATION AUDITOR v2.1 — APP ENGINE
// Full courtroom logic with RAG quote verification
// ==========================================

const API_BASE = window.location.origin;

// ==========================================
// GLOBAL STATE
// ==========================================
let selectedFile = null;
let auditData = null;
let allResults = [];
let chatHistory = [];
let useAuditContext = false;
let bulkFiles = [];

// ==========================================
// BOOT SEQUENCE
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    // Generate session ID
    const sid = 'SCI-' + Math.random().toString(36).substring(2, 8).toUpperCase();
    document.getElementById('session-id').textContent = sid;

    // Start clock
    updateClock();
    setInterval(updateClock, 1000);

    // Check server
    checkServer();
    setInterval(checkServer, 30000);

    // Dismiss oath screen after animation
    setTimeout(() => {
        const oath = document.getElementById('oath-screen');
        oath.classList.add('dismissed');
        setTimeout(() => {
            oath.style.display = 'none';
            document.getElementById('app').classList.remove('hidden');
        }, 1000);
    }, 4000);

    // Setup file upload
    setupFileUpload();
    setupBulkUpload();
    setupFilterTabs();
    setupModal();
    loadHistory();
});

function updateClock() {
    const el = document.getElementById('bench-clock');
    if (el) {
        const now = new Date();
        el.textContent = now.toLocaleTimeString('en-IN', {
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
        });
    }
}

async function checkServer() {
    try {
        const res = await fetch(`${API_BASE}/db-stats`);
        const data = await res.json();

        setIndicator('ind-api', true);
        setIndicator('ind-llm', true);
        setIndicator('ind-db', data.loaded);
        setIndicator('ind-rag', true);

        const regEl = document.getElementById('registry-records');
        if (regEl && data.record_count) {
            regEl.textContent = `${data.record_count.toLocaleString()} CASES LOADED`;
        }
    } catch (e) {
        setIndicator('ind-api', false);
        setIndicator('ind-llm', false);
        setIndicator('ind-db', false);
        setIndicator('ind-rag', false);
    }
}

function setIndicator(id, online) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('online', online);
    el.classList.toggle('offline', !online);
}

// ==========================================
// TAB NAVIGATION
// ==========================================
function switchTab(tab) {
    const tabs = ['audit', 'search', 'bulk', 'history'];
    tabs.forEach(t => {
        const el = document.getElementById(`tab-${t}`);
        const btn = document.getElementById(`nav-${t}`);
        if (el) el.classList.toggle('hidden', t !== tab);
        if (btn) btn.classList.toggle('active', t === tab);
    });
}

// ==========================================
// FILE UPLOAD
// ==========================================
function setupFileUpload() {
    const zone = document.getElementById('seal-dropzone');
    const input = document.getElementById('file-input');
    const removeBtn = document.getElementById('filed-remove');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', e => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    removeBtn.addEventListener('click', clearFile);

    document.getElementById('gavel-btn').addEventListener('click', runAudit);
}

function handleFile(file) {
    if (file.type !== 'application/pdf') {
        showToast('Only PDF files are accepted', 'error');
        return;
    }
    selectedFile = file;
    document.getElementById('seal-dropzone').classList.add('hidden');
    document.getElementById('filed-doc').classList.remove('hidden');
    document.getElementById('filed-name').textContent = file.name;
    document.getElementById('filed-size').textContent = formatSize(file.size);
    document.getElementById('gavel-btn').disabled = false;
    showToast('Document filed successfully', 'success');
}

function clearFile() {
    selectedFile = null;
    document.getElementById('seal-dropzone').classList.remove('hidden');
    document.getElementById('filed-doc').classList.add('hidden');
    document.getElementById('gavel-btn').disabled = true;
    document.getElementById('file-input').value = '';
}

function formatSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// ==========================================
// MAIN AUDIT
// ==========================================
async function runAudit() {
    if (!selectedFile) return;

    // Show deliberation
    document.getElementById('chamber-idle').classList.add('hidden');
    document.getElementById('chamber-results').classList.add('hidden');
    document.getElementById('chamber-deliberation').classList.remove('hidden');
    document.getElementById('gavel-btn').disabled = true;

    // Reset steps
    for (let i = 1; i <= 6; i++) {
        const step = document.getElementById(`ds-${i}`);
        step.classList.remove('active', 'done');
    }
    activateStep(1);

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        activateStep(2);
        const res = await fetch(`${API_BASE}/audit-document`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);

        activateStep(3);
        await sleep(500);
        activateStep(4);
        await sleep(500);
        activateStep(5);

        const data = await res.json();
        activateStep(6);
        await sleep(500);

        auditData = data;
        allResults = data.results || [];

        // Save to history
        saveToHistory(data);

        // Render
        renderVerdictSummary(data);
        renderResults(allResults);

        // Show results
        document.getElementById('chamber-deliberation').classList.add('hidden');
        document.getElementById('chamber-results').classList.remove('hidden');

        // Show post-audit actions
        document.getElementById('post-audit-actions').classList.remove('hidden');
        document.getElementById('court-breakdown').classList.remove('hidden');

        showToast('Judgment pronounced!', 'success');

    } catch (e) {
        showToast(`Audit failed: ${e.message}`, 'error');
        document.getElementById('chamber-deliberation').classList.add('hidden');
        document.getElementById('chamber-idle').classList.remove('hidden');
    }

    document.getElementById('gavel-btn').disabled = false;
}

function activateStep(n) {
    for (let i = 1; i <= 6; i++) {
        const step = document.getElementById(`ds-${i}`);
        if (i < n) {
            step.classList.remove('active');
            step.classList.add('done');
        } else if (i === n) {
            step.classList.add('active');
            step.classList.remove('done');
        } else {
            step.classList.remove('active', 'done');
        }
    }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ==========================================
// RENDER VERDICT SUMMARY
// ==========================================
function renderVerdictSummary(data) {
    const results = data.results || [];

    let verified = 0, fabricated = 0, skipped = 0, unverified = 0;
    let qVerified = 0, qContradicted = 0, qUnsupported = 0, qFabricated = 0;

    results.forEach(r => {
        const status = (r.verification || {}).status || '';
        if (status.includes('VERIFIED') && !status.includes('HC-')) verified++;
        else if (status.includes('HALLUCINATION')) fabricated++;
        else if (status.includes('SKIPPED') || status.includes('HC-')) skipped++;
        else unverified++;

        // Quote verification
        const qStatus = (r.quote_verification || {}).status || '';
        const qVerdict = (r.quote_verification || {}).verdict || '';
        if (qVerdict === 'SUPPORTED' || qStatus.includes('VERIFIED')) qVerified++;
        else if (qVerdict === 'CONTRADICTED' || qStatus.includes('CONTRADICTED')) qContradicted++;
        else if (qVerdict === 'UNSUPPORTED') qUnsupported++;
        if (qStatus.includes('SKIPPED') && status.includes('HALLUCINATION')) qFabricated++;
    });

    document.getElementById('vc-total').textContent = results.length;
    document.getElementById('vc-upheld').textContent = verified;
    document.getElementById('vc-overruled').textContent = fabricated;
    document.getElementById('vc-skipped').textContent = skipped;
    document.getElementById('vc-unheard').textContent = unverified;

    // Quote summary
    document.getElementById('vc-quote-verified').textContent = qVerified;
    document.getElementById('vc-quote-contradicted').textContent = qContradicted;
    document.getElementById('vc-quote-unsupported').textContent = qUnsupported;
    document.getElementById('vc-quote-fabricated').textContent = qFabricated;
    if (qVerified + qContradicted + qUnsupported + qFabricated > 0) {
        document.getElementById('quote-summary').classList.remove('hidden');
    }

    // Court breakdown bars
    const sc = data.supreme_court_count || 0;
    const hc = data.high_court_count || 0;
    const total = sc + hc || 1;
    document.getElementById('cb-sc-fill').style.width = `${(sc / total) * 100}%`;
    document.getElementById('cb-hc-fill').style.width = `${(hc / total) * 100}%`;
    document.getElementById('cb-sc-count').textContent = sc;
    document.getElementById('cb-hc-count').textContent = hc;

    // Filter counts
    document.getElementById('jf-all').textContent = results.length;
    document.getElementById('jf-upheld').textContent = verified;
    document.getElementById('jf-fabricated').textContent = fabricated;
    document.getElementById('jf-skipped').textContent = skipped;
    document.getElementById('jf-unverified').textContent = unverified;
    document.getElementById('jf-quote-issues').textContent = qContradicted + qUnsupported;
}

// ==========================================
// RENDER RESULT CARDS
// ==========================================
function renderResults(results, filter = 'all') {
    const roll = document.getElementById('judgment-roll');
    roll.innerHTML = '';

    const filtered = results.filter(r => {
        if (filter === 'all') return true;
        const cat = categorize(r);
        if (filter === 'quote-issue') {
            const qv = (r.quote_verification || {}).verdict || '';
            return qv === 'CONTRADICTED' || qv === 'UNSUPPORTED';
        }
        return cat === filter;
    });

    if (filtered.length === 0) {
        roll.innerHTML = '<div style="text-align:center;padding:3rem;color:var(--text-muted);"><p>No results match this filter.</p></div>';
        return;
    }

    filtered.forEach((r, i) => {
        const card = createJudgmentCard(r, i);
        roll.appendChild(card);
    });
}

function categorize(r) {
    const status = (r.verification || {}).status || '';
    if (status.includes('VERIFIED') && !status.includes('HC-')) return 'verified';
    if (status.includes('HALLUCINATION')) return 'hallucinated';
    if (status.includes('SKIPPED') || status.includes('HC-')) return 'skipped';
    return 'no-match';
}

function createJudgmentCard(r, index) {
    const cat = categorize(r);
    const v = r.verification || {};
    const q = r.quote_verification || {};
    const confidence = v.confidence || 0;

    // Verdict label
    const verdictLabels = {
        'verified': 'UPHELD',
        'hallucinated': 'FABRICATED',
        'skipped': 'HIGH COURT',
        'no-match': 'UNVERIFIED'
    };

    // Quote verification badge
    let qvBadge = '';
    const qVerdict = q.verdict || '';
    const qStatus = q.status || '';
    if (qStatus.includes('VERIFIED') || qVerdict === 'SUPPORTED') {
        qvBadge = '<span class="qv-badge qv-ok"><i class="fas fa-check"></i> Quote OK</span>';
    } else if (qVerdict === 'CONTRADICTED' || qStatus.includes('CONTRADICTED')) {
        qvBadge = '<span class="qv-badge qv-bad"><i class="fas fa-times"></i> Contradicted</span>';
    } else if (qVerdict === 'PARTIALLY_SUPPORTED') {
        qvBadge = '<span class="qv-badge qv-warn"><i class="fas fa-exclamation"></i> Partial</span>';
    } else if (qVerdict === 'UNSUPPORTED') {
        qvBadge = '<span class="qv-badge qv-warn"><i class="fas fa-question"></i> Unsupported</span>';
    } else if (qStatus.includes('SKIPPED')) {
        qvBadge = '<span class="qv-badge qv-skip"><i class="fas fa-forward"></i> Skipped</span>';
    }

    // Has quote issue?
    const hasQuoteIssue = qVerdict === 'CONTRADICTED' || qVerdict === 'UNSUPPORTED';

    // Confidence bar
    const confClass = confidence >= 70 ? 'high' : confidence >= 40 ? 'mid' : 'low';

    const card = document.createElement('div');
    card.className = `j-card ${cat} ${hasQuoteIssue ? 'quote-issue' : ''}`;
    card.style.animationDelay = `${index * 0.05}s`;
    card.onclick = () => openOrderModal(r, index);

    card.innerHTML = `
        <div class="j-card-top">
            <div class="j-case-name">${r.target_citation || 'Unknown Citation'}</div>
            <div class="j-badges">
                <span class="j-verdict-badge">${verdictLabels[cat] || 'UNKNOWN'}</span>
                ${qvBadge}
            </div>
        </div>
        ${confidence > 0 ? `
        <div class="confidence-bar-wrap">
            <span class="confidence-label">CONFIDENCE</span>
            <div class="confidence-track">
                <div class="confidence-fill ${confClass}" style="width:${confidence}%"></div>
            </div>
            <span class="confidence-pct">${confidence}%</span>
        </div>` : ''}
        <div class="j-card-details">
            ${v.matched_name ? `<div class="j-detail"><i class="fas fa-check"></i><span class="j-label">MATCH</span><span class="j-value">${v.matched_name}</span></div>` : ''}
            ${v.reason ? `<div class="j-detail"><i class="fas fa-comment"></i><span class="j-label">REASON</span><span class="j-value">${truncate(v.reason, 80)}</span></div>` : ''}
            ${v.message ? `<div class="j-detail"><i class="fas fa-info-circle"></i><span class="j-label">NOTE</span><span class="j-value">${truncate(v.message, 80)}</span></div>` : ''}
        </div>
        <div class="j-card-foot">
            <span class="j-read-order"><i class="fas fa-file-alt"></i> Read Full Order</span>
            <span class="j-serial">#${String(index + 1).padStart(3, '0')}</span>
        </div>
    `;

    return card;
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.substring(0, len) + '...' : str;
}

// ==========================================
// FILTER TABS
// ==========================================
function setupFilterTabs() {
    document.addEventListener('click', e => {
        const tab = e.target.closest('.jf-tab');
        if (!tab) return;

        document.querySelectorAll('.jf-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const filter = tab.dataset.filter;
        renderResults(allResults, filter);
    });
}

// ==========================================
// ORDER MODAL
// ==========================================
function setupModal() {
    document.getElementById('order-close').addEventListener('click', () => {
        document.getElementById('order-overlay').classList.add('hidden');
    });

    document.getElementById('order-overlay').addEventListener('click', e => {
        if (e.target === e.currentTarget) {
            e.currentTarget.classList.add('hidden');
        }
    });
}

function openOrderModal(r, index) {
    const body = document.getElementById('order-body');
    const v = r.verification || {};
    const q = r.quote_verification || {};
    const cat = categorize(r);

    const verdictText = {
        'verified': '🟢 CITATION UPHELD — Verified in Supreme Court Registry',
        'hallucinated': '🔴 CITATION FABRICATED — No matching case found',
        'skipped': '🟡 HIGH COURT CITATION — Not verified against SC registry',
        'no-match': '⚪ UNVERIFIED — Could not determine validity'
    };

    let qvSection = '';
    if (q.status && !q.status.includes('SKIPPED')) {
        qvSection = `
            <div class="order-section order-quote-section">
                <div class="order-section-title">QUOTE VERIFICATION (RAG ANALYSIS)</div>
                <div class="order-field">
                    <div class="of-label">VERDICT</div>
                    <div class="of-value" style="font-weight:700;">${q.verdict || q.status || 'N/A'}</div>
                </div>
                ${q.reason ? `<div class="order-field"><div class="of-label">ANALYSIS</div><div class="of-value">${q.reason}</div></div>` : ''}
                ${q.similarity_score ? `<div class="order-field"><div class="of-label">SIMILARITY SCORE</div><div class="of-value">${(q.similarity_score * 100).toFixed(1)}%</div></div>` : ''}
                ${q.chunks_analyzed ? `<div class="order-field"><div class="of-label">CHUNKS ANALYZED</div><div class="of-value">${q.chunks_analyzed}</div></div>` : ''}
                ${q.found_paragraph ? `
                    <div class="order-field">
                        <div class="of-label">SOURCE PARAGRAPH FROM JUDGMENT</div>
                        <div class="order-source-paragraph">${q.found_paragraph}</div>
                    </div>` : ''}
            </div>`;
    }

    body.innerHTML = `
        <div class="order-section">
            <div class="order-section-title">MATTER BEFORE THE COURT</div>
            <div class="order-field">
                <div class="of-label">CITED AUTHORITY</div>
                <div class="of-value" style="font-family:var(--font-legal);font-size:1rem;font-weight:600;">${r.target_citation || 'Unknown'}</div>
            </div>
            <div class="order-field">
                <div class="of-label">COURT CLASSIFICATION</div>
                <div class="of-value">${r.court_type || 'Unknown'}</div>
            </div>
            <div class="order-field">
                <div class="of-label">SERIAL NUMBER</div>
                <div class="of-value">#${String(index + 1).padStart(3, '0')}</div>
            </div>
        </div>

        <div class="order-verdict-banner ${cat}">
            ${verdictText[cat] || 'UNKNOWN STATUS'}
        </div>

        <div class="order-section">
            <div class="order-section-title">VERIFICATION DETAILS</div>
            ${v.matched_name ? `<div class="order-field"><div class="of-label">MATCHED CASE</div><div class="of-value">${v.matched_name}</div></div>` : ''}
            ${v.matched_citation ? `<div class="order-field"><div class="of-label">DATABASE CITATION</div><div class="of-value" style="font-family:var(--font-mono);font-size:0.8rem;">${v.matched_citation}</div></div>` : ''}
            ${v.reason ? `<div class="order-field"><div class="of-label">REASONING</div><div class="of-value">${v.reason}</div></div>` : ''}
            ${v.message ? `<div class="order-field"><div class="of-label">NOTE</div><div class="of-value">${v.message}</div></div>` : ''}
            ${v.confidence ? `<div class="order-field"><div class="of-label">CONFIDENCE</div><div class="of-value">${v.confidence}%</div></div>` : ''}
        </div>

        ${qvSection}
    `;

    document.getElementById('order-overlay').classList.remove('hidden');
}

// ==========================================
// MANUAL SEARCH
// ==========================================
function fillSearch(text) {
    document.getElementById('manual-search-input').value = text;
}

async function runManualSearch() {
    const input = document.getElementById('manual-search-input');
    const area = document.getElementById('search-results-area');
    const citation = input.value.trim();

    if (!citation) {
        showToast('Please enter a citation to verify', 'warning');
        return;
    }

    area.innerHTML = '<div class="search-loading"><i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Verifying citation...</p></div>';

    try {
        const res = await fetch(`${API_BASE}/verify-citation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ citation })
        });

        const data = await res.json();
        const v = data.verification || {};
        const status = v.status || '';

        let cat = 'no-match';
        if (status.includes('VERIFIED')) cat = 'verified';
        else if (status.includes('HALLUCINATION')) cat = 'hallucinated';

        const verdictLabels = { verified: '🟢 UPHELD', hallucinated: '🔴 FABRICATED', 'no-match': '⚪ UNVERIFIED' };

        area.innerHTML = `
            <div class="search-result-card ${cat}">
                <div class="src-header">
                    <div class="src-citation">${citation}</div>
                    <div class="src-verdict">${verdictLabels[cat] || status}</div>
                </div>
                <div class="src-details">
                    ${v.matched_name ? `<div class="src-field"><span class="src-label">MATCH</span><span>${v.matched_name}</span></div>` : ''}
                    ${v.matched_citation ? `<div class="src-field"><span class="src-label">CITATION</span><span style="font-family:var(--font-mono);">${v.matched_citation}</span></div>` : ''}
                    ${v.reason ? `<div class="src-field"><span class="src-label">REASON</span><span>${v.reason}</span></div>` : ''}
                    ${v.message ? `<div class="src-field"><span class="src-label">NOTE</span><span>${v.message}</span></div>` : ''}
                    ${v.confidence ? `<div class="src-field"><span class="src-label">CONFIDENCE</span><span>${v.confidence}%</span></div>` : ''}
                    <div class="src-field"><span class="src-label">COURT</span><span>${data.court_type || 'Unknown'}</span></div>
                </div>
            </div>
        `;
    } catch (e) {
        area.innerHTML = `<div class="search-error"><i class="fas fa-exclamation-triangle" style="font-size:2rem;"></i><p>Error: ${e.message}</p></div>`;
    }
}

// ==========================================
// BULK AUDIT
// ==========================================
function setupBulkUpload() {
    const zone = document.getElementById('bulk-dropzone');
    const input = document.getElementById('bulk-file-input');

    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        addBulkFiles(e.dataTransfer.files);
    });

    input.addEventListener('change', e => addBulkFiles(e.target.files));
}

function addBulkFiles(fileList) {
    for (const file of fileList) {
        if (file.type !== 'application/pdf') continue;
        if (bulkFiles.some(f => f.name === file.name)) continue;
        bulkFiles.push(file);
    }
    renderBulkFileList();
    document.getElementById('bulk-audit-btn').disabled = bulkFiles.length === 0;
}

function renderBulkFileList() {
    const list = document.getElementById('bulk-file-list');
    list.innerHTML = bulkFiles.map((f, i) => `
        <div class="bulk-file-item">
            <i class="fas fa-file-pdf" style="color:var(--red);"></i>
            <span class="bfi-name">${f.name}</span>
            <span class="bfi-size">${formatSize(f.size)}</span>
            <button class="bfi-remove" onclick="removeBulkFile(${i})"><i class="fas fa-times"></i></button>
        </div>
    `).join('');
}

function removeBulkFile(index) {
    bulkFiles.splice(index, 1);
    renderBulkFileList();
    document.getElementById('bulk-audit-btn').disabled = bulkFiles.length === 0;
}

async function runBulkAudit() {
    if (bulkFiles.length === 0) return;

    const progress = document.getElementById('bulk-progress');
    const fill = document.getElementById('bulk-progress-fill');
    const text = document.getElementById('bulk-progress-text');
    const area = document.getElementById('bulk-results-area');
    const btn = document.getElementById('bulk-audit-btn');

    progress.classList.remove('hidden');
    btn.disabled = true;
    fill.style.width = '10%';
    text.textContent = 'Uploading documents...';

    try {
        const formData = new FormData();
        bulkFiles.forEach(f => formData.append('files', f));

        fill.style.width = '30%';
        text.textContent = 'Processing citations...';

        const res = await fetch(`${API_BASE}/audit-multiple`, {
            method: 'POST',
            body: formData
        });

        fill.style.width = '80%';
        text.textContent = 'Analyzing results...';

        const data = await res.json();

        fill.style.width = '100%';
        text.textContent = 'Complete!';

        // Count quote issues
        let totalQuoteIssues = 0;
        (data.documents || []).forEach(doc => {
            (doc.results || []).forEach(r => {
                const qv = (r.quote_verification || {}).verdict || '';
                if (qv === 'CONTRADICTED' || qv === 'UNSUPPORTED') totalQuoteIssues++;
            });
        });

        area.innerHTML = `
            <div class="bulk-summary-header">
                <div class="bulk-stat"><div class="bs-num">${data.total_documents || 0}</div><div class="bs-label">DOCUMENTS</div></div>
                <div class="bulk-stat"><div class="bs-num">${(data.total_sc_citations || 0) + (data.total_hc_citations || 0)}</div><div class="bs-label">CITATIONS</div></div>
                <div class="bulk-stat upheld"><div class="bs-num">${data.total_verified || 0}</div><div class="bs-label">UPHELD</div></div>
                <div class="bulk-stat fabricated"><div class="bs-num">${data.total_fabricated || 0}</div><div class="bs-label">FABRICATED</div></div>
                <div class="bulk-stat quote-issues"><div class="bs-num">${totalQuoteIssues}</div><div class="bs-label">QUOTE ISSUES</div></div>
            </div>
            ${(data.documents || []).map(doc => {
                const hasError = doc.error;
                return `
                <div class="bulk-doc-card ${hasError ? 'error' : ''}">
                    <div class="bdc-header">
                        <i class="fas fa-file-pdf" style="color:${hasError ? 'var(--red)' : 'var(--gold)'};"></i>
                        <span class="bdc-name">${doc.filename}</span>
                        <span class="bdc-badge ${hasError ? 'error' : ''}">${hasError ? 'ERROR' : `${doc.citations_found || 0} citations`}</span>
                    </div>
                    ${hasError ? `<p style="font-size:0.7rem;color:var(--red);margin-top:0.3rem;">${doc.error}</p>` : `
                    <div class="bdc-stats">
                        <span>SC: ${doc.sc_count || 0}</span>
                        <span>HC: ${doc.hc_count || 0}</span>
                        <span style="color:var(--green);">Upheld: ${(doc.results || []).filter(r => (r.verification||{}).status?.includes('VERIFIED')).length}</span>
                        <span style="color:var(--red);">Fabricated: ${(doc.results || []).filter(r => (r.verification||{}).status?.includes('HALLUCINATION')).length}</span>
                    </div>`}
                </div>`;
            }).join('')}
        `;

        showToast('Bulk audit complete!', 'success');
    } catch (e) {
        showToast(`Bulk audit failed: ${e.message}`, 'error');
        area.innerHTML = `<div class="search-error"><p>Error: ${e.message}</p></div>`;
    }

    btn.disabled = false;
}

// ==========================================
// AI SUMMARY
// ==========================================
async function generateSummary() {
    if (!auditData) return showToast('Run an audit first', 'warning');

    const overlay = document.getElementById('summary-overlay');
    const body = document.getElementById('summary-body');
    overlay.classList.remove('hidden');
    body.innerHTML = '<div class="summary-loading"><i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Generating professional summary...</p></div>';

    try {
        const res = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: auditData.results || [],
                total: auditData.total_citations_found || 0,
                sc_count: auditData.supreme_court_count || 0,
                hc_count: auditData.high_court_count || 0
            })
        });

        const data = await res.json();

        body.innerHTML = `
            <div class="order-section">
                <div class="order-section-title">RISK ASSESSMENT</div>
                <div class="order-verdict-banner ${data.risk_level === 'High' ? 'hallucinated' : data.risk_level === 'Medium' ? 'skipped' : 'verified'}">
                    RISK LEVEL: ${data.risk_level || 'Unknown'}
                </div>
            </div>
            <div class="order-section">
                <div class="order-section-title">PROFESSIONAL SUMMARY</div>
                <div class="of-value" style="white-space:pre-wrap;line-height:1.8;">${data.summary || 'No summary available.'}</div>
            </div>
            <div class="order-section">
                <div class="order-section-title">STATISTICS</div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;">
                    <div class="verdict-card upheld"><div class="vc-number">${data.stats?.verified || 0}</div><div class="vc-label">VERIFIED</div></div>
                    <div class="verdict-card overruled"><div class="vc-number">${data.stats?.fabricated || 0}</div><div class="vc-label">FABRICATED</div></div>
                    <div class="verdict-card skipped"><div class="vc-number">${data.stats?.skipped || 0}</div><div class="vc-label">SKIPPED</div></div>
                    <div class="verdict-card unheard"><div class="vc-number">${data.stats?.unverified || 0}</div><div class="vc-label">UNVERIFIED</div></div>
                </div>
            </div>
        `;
    } catch (e) {
        body.innerHTML = `<div class="search-error"><p>Error: ${e.message}</p></div>`;
    }
}

// ==========================================
// EXPORT
// ==========================================
function exportCSV() {
    if (!allResults.length) return showToast('No results to export', 'warning');

    let csv = 'Citation,Court Type,Verification Status,Confidence,Matched Case,Quote Verdict,Quote Reason\n';
    allResults.forEach(r => {
        const v = r.verification || {};
        const q = r.quote_verification || {};
        csv += `"${(r.target_citation || '').replace(/"/g, '""')}","${r.court_type || ''}","${v.status || ''}","${v.confidence || ''}","${(v.matched_name || '').replace(/"/g, '""')}","${q.verdict || q.status || ''}","${(q.reason || '').replace(/"/g, '""')}"\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'citation_audit_report.csv';
    a.click();
    URL.revokeObjectURL(url);
    showToast('CSV exported', 'success');
}

function exportPDF() {
    if (!allResults.length) return showToast('No results to export', 'warning');

    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        doc.setFontSize(16);
        doc.text('Legal Citation Audit Report', 20, 20);
        doc.setFontSize(10);
        doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 28);
        doc.text(`Total Citations: ${allResults.length}`, 20, 34);

        let y = 45;
        allResults.forEach((r, i) => {
            if (y > 270) { doc.addPage(); y = 20; }
            const v = r.verification || {};
            const q = r.quote_verification || {};
            doc.setFontSize(9);
            doc.text(`${i + 1}. ${(r.target_citation || 'Unknown').substring(0, 80)}`, 20, y);
            y += 5;
            doc.setFontSize(7);
            doc.text(`   Status: ${v.status || 'Unknown'} | Confidence: ${v.confidence || 'N/A'}% | Quote: ${q.verdict || q.status || 'N/A'}`, 20, y);
            y += 8;
        });

        doc.save('citation_audit_report.pdf');
        showToast('PDF exported', 'success');
    } catch (e) {
        showToast('PDF export failed: ' + e.message, 'error');
    }
}

function exportSummaryPDF() {
    showToast('Summary download coming soon', 'info');
}

// ==========================================
// HISTORY
// ==========================================
function saveToHistory(data) {
    const history = JSON.parse(localStorage.getItem('audit_history') || '[]');
    history.unshift({
        filename: data.filename || selectedFile?.name || 'Unknown',
        date: new Date().toISOString(),
        total: data.total_citations_found || 0,
        sc: data.supreme_court_count || 0,
        hc: data.high_court_count || 0,
        verified: (data.results || []).filter(r => (r.verification || {}).status?.includes('VERIFIED')).length,
        fabricated: (data.results || []).filter(r => (r.verification || {}).status?.includes('HALLUCINATION')).length,
    });

    // Keep last 50
    if (history.length > 50) history.length = 50;
    localStorage.setItem('audit_history', JSON.stringify(history));
    loadHistory();
}

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('audit_history') || '[]');
    const list = document.getElementById('history-list');
    const count = document.getElementById('history-count');

    count.textContent = `${history.length} records`;

    if (history.length === 0) {
        list.innerHTML = '<div class="history-empty"><i class="fas fa-history" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i><p>No audit history yet.</p></div>';
        return;
    }

    list.innerHTML = history.map(h => `
        <div class="history-item">
            <div class="hi-top">
                <span class="hi-file"><i class="fas fa-file-pdf" style="color:var(--red);margin-right:0.3rem;"></i>${h.filename}</span>
                <span class="hi-date">${new Date(h.date).toLocaleDateString('en-IN')}</span>
            </div>
            <div class="hi-stats">
                <span>Total: ${h.total}</span>
                <span>SC: ${h.sc}</span>
                <span>HC: ${h.hc}</span>
                <span style="color:var(--green);">✓ ${h.verified}</span>
                <span style="color:var(--red);">✗ ${h.fabricated}</span>
            </div>
        </div>
    `).join('');
}

function clearHistory() {
    localStorage.removeItem('audit_history');
    loadHistory();
    showToast('History cleared', 'info');
}

// ==========================================
// TOAST NOTIFICATIONS
// ==========================================
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const icons = { success: 'fa-check-circle', error: 'fa-times-circle', warning: 'fa-exclamation-triangle', info: 'fa-info-circle' };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${message}</span>`;
    container.appendChild(toast);

    requestAnimationFrame(() => toast.classList.add('show'));

    setTimeout(() => {
        toast.classList.remove('show');
        toast.classList.add('hide');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==========================================
// CHATBOT
// ==========================================
function toggleChat() {
    const pane = document.getElementById('chat-pane');
    pane.classList.toggle('hidden');
    document.getElementById('chat-notification').style.display = 'none';
}

function toggleAuditContext() {
    useAuditContext = !useAuditContext;
    const btn = document.getElementById('ctx-btn');
    const bar = document.getElementById('chat-context-bar');
    btn.classList.toggle('active', useAuditContext);
    bar.classList.toggle('hidden', !useAuditContext);
}

function clearChat() {
    chatHistory = [];
    const msgs = document.getElementById('chat-messages');
    msgs.innerHTML = `
        <div class="chat-msg assistant">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble"><p>Chat cleared. How can I help you?</p></div>
        </div>
    `;
}

function handleChatKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
}

function autoResizeChat(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

function sendSuggestion(text) {
    document.getElementById('chat-input').value = text;
    sendChatMessage();
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const msgs = document.getElementById('chat-messages');
    const message = input.value.trim();
    if (!message) return;

    // Add user message
    msgs.innerHTML += `
        <div class="chat-msg user">
            <div class="msg-avatar">👤</div>
            <div class="msg-bubble"><p>${escapeHtml(message)}</p></div>
        </div>
    `;

    input.value = '';
    input.style.height = 'auto';
    document.getElementById('chat-suggestions').style.display = 'none';

    // Typing indicator
    const typingId = 'typing-' + Date.now();
    msgs.innerHTML += `
        <div class="chat-msg assistant" id="${typingId}">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble typing-indicator"><span></span><span></span><span></span></div>
        </div>
    `;
    msgs.scrollTop = msgs.scrollHeight;

    // Build context
    let auditContext = null;
    if (useAuditContext && auditData) {
        const verified = (auditData.results || []).filter(r => (r.verification || {}).status?.includes('VERIFIED')).length;
        const fabricated = (auditData.results || []).filter(r => (r.verification || {}).status?.includes('HALLUCINATION')).length;
        auditContext = `Audit of "${auditData.filename}": ${auditData.total_citations_found} citations, ${verified} verified, ${fabricated} fabricated.`;
    }

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                history: chatHistory,
                audit_context: auditContext
            })
        });

        const data = await res.json();

        // Remove typing
        document.getElementById(typingId)?.remove();

        // Add response
        msgs.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-avatar">⚖️</div>
                <div class="msg-bubble"><p>${formatChat(data.reply)}</p></div>
            </div>
        `;

        chatHistory.push({ role: 'user', content: message });
        chatHistory.push({ role: 'assistant', content: data.reply });
        if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);

    } catch (e) {
        document.getElementById(typingId)?.remove();
        msgs.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-avatar">⚖️</div>
                <div class="msg-bubble"><p style="color:var(--red);">Error: ${e.message}</p></div>
            </div>
        `;
    }

    msgs.scrollTop = msgs.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatChat(text) {
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/^[-•]\s+(.+)$/gm, '<br>• $1');
    html = html.replace(/\n/g, '<br>');
    return html;
}
```

---

### File: `frontend/static/styles.css`

```css
/* ==========================================
   ⚖️ LEGAL CITATION AUDITOR v2.1 — STYLES
   Complete stylesheet with RAG quote verification support
   ========================================== */

/* ===== CSS VARIABLES ===== */
:root {
    --bg-deep: #0a0a0f;
    --bg-panel: #111118;
    --bg-card: #16161f;
    --bg-elevated: #1c1c28;
    --bg-hover: #222230;
    --gold: #c9a84c;
    --gold-light: #e0c878;
    --gold-dark: #9a7b30;
    --gold-glow: rgba(201, 168, 76, 0.15);
    --green: #4caf8a;
    --green-glow: rgba(76, 175, 138, 0.15);
    --red: #e87777;
    --red-glow: rgba(232, 119, 119, 0.15);
    --amber: #e8b877;
    --amber-glow: rgba(232, 184, 119, 0.15);
    --blue: #77b8e8;
    --blue-glow: rgba(119, 184, 232, 0.15);
    --purple: #b877e8;
    --purple-glow: rgba(184, 119, 232, 0.15);
    --text-primary: #e8e4dd;
    --text-secondary: #8a8578;
    --text-muted: #5a564f;
    --border: rgba(201, 168, 76, 0.12);
    --border-strong: rgba(201, 168, 76, 0.25);
    --font-display: 'Playfair Display', serif;
    --font-body: 'Inter', sans-serif;
    --font-legal: 'Cormorant Garamond', serif;
    --font-mono: 'JetBrains Mono', monospace;
    --radius: 8px;
    --radius-lg: 12px;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 8px 48px rgba(0, 0, 0, 0.6);
    --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== RESET ===== */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: var(--font-body);
    background: var(--bg-deep);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
    line-height: 1.6;
}

.hidden { display: none !important; }

/* ===== OATH SCREEN ===== */
.oath-screen {
    position: fixed; inset: 0; z-index: 9999;
    background: radial-gradient(ellipse at center, #111118 0%, #0a0a0f 70%);
    display: flex; align-items: center; justify-content: center;
    transition: opacity 1s ease, transform 1s ease;
}
.oath-screen.dismissed { opacity: 0; transform: scale(1.05); pointer-events: none; }
.oath-content { text-align: center; max-width: 500px; padding: 2rem; }

.oath-emblem { position: relative; width: 100px; height: 100px; margin: 0 auto 1.5rem; }
.emblem-outer-ring, .emblem-inner-ring {
    position: absolute; border-radius: 50%; border: 2px solid var(--gold);
}
.emblem-outer-ring { inset: 0; animation: spin 12s linear infinite; opacity: 0.3; }
.emblem-inner-ring { inset: 10px; animation: spin 8s linear infinite reverse; opacity: 0.5; }
.emblem-icon { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 2.5rem; }
@keyframes spin { to { transform: rotate(360deg); } }

.oath-heading {
    font-family: var(--font-display); font-size: 1.6rem; font-weight: 700;
    color: var(--gold); letter-spacing: 0.15em; margin-bottom: 0.25rem;
}
.oath-subheading {
    font-family: var(--font-legal); font-size: 0.95rem; color: var(--text-secondary);
    font-style: italic; margin-bottom: 1.5rem;
}
.oath-divider { display: flex; align-items: center; gap: 0.5rem; justify-content: center; margin-bottom: 1.5rem; }
.divider-wing { flex: 1; max-width: 80px; height: 1px; background: linear-gradient(90deg, transparent, var(--gold), transparent); }
.divider-diamond { color: var(--gold); font-size: 0.6rem; }

.oath-lines { text-align: left; margin-bottom: 1.5rem; }
.oath-line {
    display: flex; align-items: center; gap: 0.75rem; padding: 0.4rem 0;
    font-size: 0.85rem; color: var(--text-secondary);
    opacity: 0; animation: fadeInLine 0.5s forwards;
    animation-delay: var(--delay);
}
@keyframes fadeInLine { to { opacity: 1; } }
.oath-bullet { color: var(--gold); font-weight: 700; font-size: 0.9rem; }
.oath-check { color: var(--green); margin-left: auto; opacity: 0; animation: checkPop 0.3s forwards; animation-delay: calc(var(--delay) + 0.4s); }
@keyframes checkPop { to { opacity: 1; transform: scale(1.2); } }

.oath-progress { margin-bottom: 1rem; }
.oath-progress-track { height: 3px; background: var(--bg-elevated); border-radius: 3px; overflow: hidden; }
.oath-progress-fill { height: 100%; background: linear-gradient(90deg, var(--gold-dark), var(--gold), var(--gold-light)); width: 0; animation: progressFill 3.5s ease forwards; }
@keyframes progressFill { to { width: 100%; } }

.oath-footer-text {
    font-family: var(--font-legal); font-size: 0.8rem; color: var(--text-muted);
    font-style: italic; letter-spacing: 0.05em;
}

/* ===== APP LAYOUT ===== */
.app { display: flex; flex-direction: column; min-height: 100vh; animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* ===== BENCH BAR (HEADER) ===== */
.bench-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 1.5rem; background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.bench-left { display: flex; align-items: center; gap: 0.75rem; }
.bench-emblem { font-size: 1.5rem; }
.bench-title-block { }
.bench-title {
    font-family: var(--font-display); font-size: 1rem; font-weight: 700;
    color: var(--gold); letter-spacing: 0.1em;
}
.version-badge {
    font-family: var(--font-mono); font-size: 0.55rem; padding: 0.1rem 0.3rem;
    background: var(--gold-glow); border: 1px solid var(--border-strong);
    border-radius: 3px; color: var(--gold); vertical-align: middle;
}
.bench-subtitle { font-size: 0.65rem; color: var(--text-secondary); margin-top: 0.1rem; }

.bench-center { display: flex; align-items: center; }
.bench-status-row { display: flex; gap: 1rem; }
.bench-indicator {
    display: flex; align-items: center; gap: 0.35rem; font-size: 0.65rem;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em;
}
.indicator-lamp {
    width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted);
}
.bench-indicator.online .indicator-lamp { background: var(--green); box-shadow: 0 0 6px var(--green); }
.bench-indicator.offline .indicator-lamp { background: var(--red); box-shadow: 0 0 6px var(--red); }

.bench-right { display: flex; align-items: center; gap: 1rem; }
.bench-nav-tabs { display: flex; gap: 0.25rem; }
.nav-tab {
    background: none; border: 1px solid transparent; color: var(--text-muted);
    padding: 0.35rem 0.75rem; border-radius: var(--radius); cursor: pointer;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em;
    transition: var(--transition); font-family: var(--font-body);
}
.nav-tab:hover { color: var(--text-primary); background: var(--bg-hover); }
.nav-tab.active {
    color: var(--gold); border-color: var(--border-strong);
    background: var(--gold-glow);
}

.bench-clock {
    font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted);
    letter-spacing: 0.05em;
}
.bench-session { display: flex; flex-direction: column; align-items: center; }
.session-label { font-size: 0.5rem; color: var(--text-muted); letter-spacing: 0.1em; }
.session-id { font-family: var(--font-mono); font-size: 0.65rem; color: var(--gold); }

/* ===== COURTROOM (Main Content) ===== */
.courtroom {
    display: flex; gap: 1rem; padding: 1rem 1.5rem; flex: 1;
}

.court-panel {
    background: var(--bg-panel); border: 1px solid var(--border);
    border-radius: var(--radius-lg); overflow: hidden;
    display: flex; flex-direction: column;
}
.filing-desk { flex: 0 0 380px; }
.judgment-chamber { flex: 1; }

.panel-title-bar {
    display: flex; align-items: center; justify-content: center; gap: 0.75rem;
    padding: 0.75rem 1rem; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg-panel) 100%);
}
.panel-title-bar h2 {
    font-family: var(--font-display); font-size: 0.85rem; font-weight: 600;
    color: var(--gold); letter-spacing: 0.12em;
}
.title-bar-ornament { color: var(--gold); opacity: 0.3; font-size: 0.7rem; letter-spacing: 0.1em; }

/* ===== SEAL DROP ZONE ===== */
.seal-dropzone {
    margin: 1rem; padding: 2rem 1rem; text-align: center;
    border: 2px dashed var(--border-strong); border-radius: var(--radius-lg);
    cursor: pointer; transition: var(--transition);
    background: linear-gradient(135deg, rgba(201,168,76,0.02) 0%, transparent 100%);
}
.seal-dropzone:hover, .seal-dropzone.drag-over {
    border-color: var(--gold); background: var(--gold-glow);
    transform: translateY(-2px);
}
.seal-visual { margin-bottom: 1rem; }
.wax-seal {
    position: relative; width: 70px; height: 70px; margin: 0 auto;
}
.seal-ring {
    position: absolute; border-radius: 50%; border: 1.5px solid var(--gold);
}
.seal-ring.r1 { inset: 0; opacity: 0.2; animation: spin 15s linear infinite; }
.seal-ring.r2 { inset: 8px; opacity: 0.35; animation: spin 10s linear infinite reverse; }
.seal-ring.r3 { inset: 16px; opacity: 0.5; }
.seal-center {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; color: var(--gold);
}
.seal-rays { display: none; }

.seal-dropzone h3 {
    font-family: var(--font-display); font-size: 0.9rem; color: var(--gold);
    letter-spacing: 0.08em; margin-bottom: 0.25rem;
}
.seal-dropzone p { font-size: 0.75rem; color: var(--text-secondary); }
.format-tags { display: flex; gap: 0.5rem; justify-content: center; margin-top: 0.75rem; }
.f-tag {
    font-size: 0.6rem; padding: 0.15rem 0.5rem; border: 1px solid var(--border);
    border-radius: 3px; color: var(--text-muted); letter-spacing: 0.06em;
}

/* ===== FILED DOCUMENT ===== */
.filed-doc {
    margin: 1rem; padding: 0.75rem; background: var(--bg-elevated);
    border: 1px solid var(--border-strong); border-radius: var(--radius);
}
.filed-header { display: flex; align-items: center; gap: 0.75rem; }
.filed-icon { font-size: 1.5rem; color: var(--gold); }
.filed-info { flex: 1; display: flex; flex-direction: column; }
.filed-name { font-size: 0.8rem; font-weight: 600; color: var(--text-primary); word-break: break-all; }
.filed-size { font-size: 0.65rem; color: var(--text-secondary); font-family: var(--font-mono); }
.filed-remove {
    background: none; border: none; color: var(--text-muted); cursor: pointer;
    padding: 0.25rem; font-size: 0.8rem; transition: var(--transition);
}
.filed-remove:hover { color: var(--red); }
.filed-stamp {
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    margin-top: 0.5rem; padding: 0.3rem; background: var(--green-glow);
    border: 1px solid rgba(76,175,138,0.2); border-radius: 4px;
    font-size: 0.6rem; color: var(--green); letter-spacing: 0.1em; font-weight: 600;
}

/* ===== GAVEL BUTTON ===== */
.gavel-btn {
    position: relative; margin: 1rem; padding: 0.85rem 1.5rem;
    background: linear-gradient(135deg, var(--gold-dark) 0%, var(--gold) 50%, var(--gold-light) 100%);
    border: none; border-radius: var(--radius); cursor: pointer;
    overflow: hidden; transition: var(--transition);
}
.gavel-btn:disabled { opacity: 0.4; cursor: not-allowed; filter: grayscale(0.5); }
.gavel-btn:not(:disabled):hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(201,168,76,0.3); }
.gavel-btn:not(:disabled):active { transform: translateY(0); }
.gavel-bg { position: absolute; inset: 0; }
.gavel-content {
    position: relative; display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    font-family: var(--font-display); font-size: 0.85rem; font-weight: 700;
    color: var(--bg-deep); letter-spacing: 0.08em;
}
.gavel-shine {
    position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    animation: shine 3s infinite;
}
@keyframes shine { 0% { left: -100%; } 50%, 100% { left: 100%; } }

/* ===== VERDICT SUMMARY ===== */
.verdict-summary {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(65px, 1fr));
    gap: 0.4rem; margin: 0 1rem 0.5rem;
}
.verdict-card {
    text-align: center; padding: 0.6rem 0.3rem;
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius); transition: var(--transition);
}
.verdict-card:hover { border-color: var(--border-strong); }
.vc-icon { font-size: 0.9rem; margin-bottom: 0.2rem; }
.vc-number { font-family: var(--font-display); font-size: 1.3rem; font-weight: 700; color: var(--text-primary); }
.vc-label { font-size: 0.5rem; color: var(--text-muted); letter-spacing: 0.08em; margin-top: 0.1rem; }

.verdict-card.total .vc-icon { color: var(--gold); }
.verdict-card.upheld .vc-icon, .verdict-card.upheld .vc-number { color: var(--green); }
.verdict-card.overruled .vc-icon, .verdict-card.overruled .vc-number { color: var(--red); }
.verdict-card.skipped .vc-icon, .verdict-card.skipped .vc-number { color: var(--amber); }
.verdict-card.unheard .vc-icon, .verdict-card.unheard .vc-number { color: var(--text-muted); }

/* Quote verification summary */
.quote-summary {
    border-top: 1px solid var(--border);
    padding-top: 0.5rem;
}
.quote-summary-title {
    grid-column: 1 / -1;
    font-family: var(--font-display);
    font-size: 0.7rem;
    color: var(--gold);
    letter-spacing: 0.1em;
    text-align: center;
    padding-bottom: 0.25rem;
}
.verdict-card.quote-verified .vc-icon, .verdict-card.quote-verified .vc-number { color: var(--green); }
.verdict-card.quote-contradicted .vc-icon, .verdict-card.quote-contradicted .vc-number { color: var(--red); }
.verdict-card.quote-unsupported .vc-icon, .verdict-card.quote-unsupported .vc-number { color: var(--amber); }
.verdict-card.quote-fabricated .vc-icon, .verdict-card.quote-fabricated .vc-number { color: var(--red); }

/* ===== COURT BREAKDOWN ===== */
.court-breakdown { margin: 0 1rem 0.5rem; padding: 0.75rem; background: var(--bg-elevated); border-radius: var(--radius); border: 1px solid var(--border); }
.cb-title { display: flex; align-items: center; gap: 0.5rem; font-size: 0.7rem; color: var(--gold); letter-spacing: 0.08em; margin-bottom: 0.5rem; }
.cb-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem; }
.cb-bar-label { font-size: 0.65rem; color: var(--text-secondary); width: 90px; }
.cb-bar-track { flex: 1; height: 6px; background: var(--bg-deep); border-radius: 3px; overflow: hidden; }
.cb-bar-fill { height: 100%; border-radius: 3px; transition: width 0.8s ease; }
.sc-fill { background: linear-gradient(90deg, var(--gold-dark), var(--gold)); }
.hc-fill { background: linear-gradient(90deg, #6a5a2a, var(--amber)); }
.cb-bar-count { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-primary); width: 25px; text-align: right; }

/* ===== POST AUDIT ACTIONS ===== */
.post-audit-actions { display: flex; gap: 0.5rem; margin: 0 1rem 1rem; flex-wrap: wrap; }
.action-btn {
    flex: 1; min-width: 100px; padding: 0.5rem 0.75rem;
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius); color: var(--text-primary); cursor: pointer;
    font-size: 0.7rem; font-weight: 600; transition: var(--transition);
    display: flex; align-items: center; justify-content: center; gap: 0.35rem;
    font-family: var(--font-body);
}
.action-btn:hover { border-color: var(--gold); background: var(--gold-glow); color: var(--gold); }

/* ===== CHAMBER STATES ===== */
.chamber-idle, .chamber-deliberation {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding: 2rem;
}
.chamber-idle h3, .chamber-deliberation h3 {
    font-family: var(--font-display); font-size: 1.1rem; color: var(--gold);
    margin-bottom: 0.5rem;
}
.chamber-idle p, .chamber-deliberation p { font-size: 0.8rem; color: var(--text-secondary); }

.idle-rag-note {
    margin-top: 1rem; padding: 0.4rem 0.8rem;
    background: var(--purple-glow); border: 1px solid rgba(184,119,232,0.2);
    border-radius: var(--radius); font-size: 0.7rem; color: var(--purple);
}

/* Scales Animation */
.idle-scales-container { margin-bottom: 1.5rem; }
.scales-beam { position: relative; text-align: center; }
.scales-pivot { font-size: 3rem; }
.beam-line { display: none; }
.scale-pan { display: none; }

/* Deliberation */
.quill-animation { margin-bottom: 1.5rem; }
.quill-body { font-size: 2.5rem; color: var(--gold); animation: quillWrite 1.5s ease infinite; }
@keyframes quillWrite { 0%, 100% { transform: rotate(-5deg); } 50% { transform: rotate(5deg); } }
.ink-drops { display: flex; justify-content: center; gap: 0.3rem; margin-top: 0.5rem; }
.ink-drop {
    width: 4px; height: 4px; background: var(--gold); border-radius: 50%;
    animation: inkDrop 1.5s ease infinite;
    animation-delay: calc(var(--d) * 0.3s);
}
@keyframes inkDrop { 0% { opacity: 0; transform: translateY(-10px); } 50% { opacity: 1; } 100% { opacity: 0; transform: translateY(10px); } }

.delib-steps { margin-top: 1.5rem; width: 100%; max-width: 350px; }
.d-step {
    display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0.75rem;
    margin-bottom: 0.35rem; border-radius: var(--radius);
    font-size: 0.75rem; color: var(--text-muted);
    transition: var(--transition);
}
.d-step.active { background: var(--gold-glow); color: var(--gold); }
.d-step.done { color: var(--green); }
.d-step-marker {
    width: 22px; height: 22px; display: flex; align-items: center; justify-content: center;
    border: 1px solid var(--border); border-radius: 50%; font-size: 0.55rem;
    font-family: var(--font-legal); font-weight: 600;
}
.d-step.active .d-step-marker { border-color: var(--gold); color: var(--gold); background: var(--gold-glow); }
.d-step.done .d-step-marker { border-color: var(--green); color: var(--green); background: var(--green-glow); }

/* ===== JUDGMENT RESULTS ===== */
.chamber-results { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

.judgment-filters {
    display: flex; gap: 0.25rem; padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border); overflow-x: auto;
    flex-wrap: wrap;
}
.jf-tab {
    background: none; border: 1px solid transparent; color: var(--text-muted);
    padding: 0.3rem 0.6rem; border-radius: var(--radius); cursor: pointer;
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.04em;
    transition: var(--transition); white-space: nowrap; font-family: var(--font-body);
}
.jf-tab:hover { color: var(--text-primary); background: var(--bg-hover); }
.jf-tab.active { color: var(--gold); border-color: var(--border-strong); background: var(--gold-glow); }
.jf-count {
    font-family: var(--font-mono); font-size: 0.6rem; margin-left: 0.3rem;
    padding: 0.05rem 0.3rem; background: var(--bg-deep); border-radius: 3px;
}

.judgment-roll { flex: 1; overflow-y: auto; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; }

/* ===== JUDGMENT CARDS ===== */
.j-card {
    padding: 0.75rem; background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius); cursor: pointer; transition: var(--transition);
    animation: cardSlideIn 0.4s ease forwards; opacity: 0;
    border-left: 3px solid var(--border);
}
@keyframes cardSlideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.j-card:hover { border-color: var(--border-strong); background: var(--bg-hover); transform: translateX(3px); }
.j-card.verified { border-left-color: var(--green); }
.j-card.hallucinated { border-left-color: var(--red); }
.j-card.skipped { border-left-color: var(--amber); }
.j-card.no-match { border-left-color: var(--text-muted); }
.j-card.quote-issue { box-shadow: inset 0 -2px 0 var(--purple); }

.j-card-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.5rem; }
.j-case-name { font-family: var(--font-legal); font-size: 0.9rem; font-weight: 600; color: var(--text-primary); flex: 1; line-height: 1.3; }

.j-badges { display: flex; flex-direction: column; gap: 0.2rem; align-items: flex-end; flex-shrink: 0; }

.j-verdict-badge {
    font-size: 0.55rem; padding: 0.15rem 0.5rem; border-radius: 3px;
    font-weight: 700; letter-spacing: 0.06em;
}
.verified .j-verdict-badge { background: var(--green-glow); color: var(--green); border: 1px solid rgba(76,175,138,0.3); }
.hallucinated .j-verdict-badge { background: var(--red-glow); color: var(--red); border: 1px solid rgba(232,119,119,0.3); }
.skipped .j-verdict-badge { background: var(--amber-glow); color: var(--amber); border: 1px solid rgba(232,184,119,0.3); }
.no-match .j-verdict-badge { background: rgba(170,170,170,0.1); color: #aaa; border: 1px solid rgba(170,170,170,0.2); }

/* Quote verification badges */
.qv-badge {
    font-size: 0.5rem; padding: 0.1rem 0.4rem; border-radius: 3px;
    font-weight: 600; letter-spacing: 0.04em; display: flex; align-items: center; gap: 0.2rem;
}
.qv-ok { background: var(--green-glow); color: var(--green); border: 1px solid rgba(76,175,138,0.2); }
.qv-bad { background: var(--red-glow); color: var(--red); border: 1px solid rgba(232,119,119,0.2); }
.qv-warn { background: var(--amber-glow); color: var(--amber); border: 1px solid rgba(232,184,119,0.2); }
.qv-skip { background: rgba(170,170,170,0.05); color: var(--text-muted); border: 1px solid rgba(170,170,170,0.15); }

.confidence-bar-wrap { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
.confidence-label { font-size: 0.6rem; color: var(--text-muted); white-space: nowrap; }
.confidence-track { flex: 1; height: 4px; background: var(--bg-deep); border-radius: 2px; overflow: hidden; }
.confidence-fill { height: 100%; border-radius: 2px; transition: width 0.8s ease; }
.confidence-fill.high { background: var(--green); }
.confidence-fill.mid { background: var(--amber); }
.confidence-fill.low { background: var(--red); }
.confidence-pct { font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-primary); }

.j-card-details { display: flex; flex-direction: column; gap: 0.3rem; }
.j-detail {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.7rem; color: var(--text-secondary);
}
.j-detail i { font-size: 0.6rem; color: var(--gold); width: 14px; text-align: center; }
.j-label { font-weight: 600; color: var(--text-muted); font-size: 0.55rem; letter-spacing: 0.06em; min-width: 40px; }
.j-value { color: var(--text-secondary); }

.j-claim-preview { background: var(--bg-deep); padding: 0.3rem 0.5rem; border-radius: 4px; margin-top: 0.1rem; }
.j-claim-preview .j-value { font-style: italic; color: var(--text-secondary); font-size: 0.65rem; }

.j-card-foot {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px solid var(--border);
}
.j-read-order { font-size: 0.6rem; color: var(--gold); display: flex; align-items: center; gap: 0.3rem; }
.j-serial { font-family: var(--font-mono); font-size: 0.55rem; color: var(--text-muted); }

/* ===== ORDER MODAL ===== */
.order-overlay {
    position: fixed; inset: 0; z-index: 1000;
    background: rgba(0,0,0,0.7); backdrop-filter: blur(5px);
    display: flex; align-items: center; justify-content: center;
    animation: fadeIn 0.2s ease;
}
.order-sheet {
    background: var(--bg-panel); border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg); max-width: 700px; width: 95%;
    max-height: 85vh; display: flex; flex-direction: column;
    box-shadow: var(--shadow-lg);
}
.order-header {
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    padding: 1rem; border-bottom: 1px solid var(--border); position: relative;
    flex-wrap: wrap;
}
.order-header h3 {
    font-family: var(--font-display); font-size: 0.9rem; color: var(--gold);
    letter-spacing: 0.12em;
}
.order-header-ornament { display: flex; align-items: center; gap: 0.3rem; }
.oh-line { display: inline-block; width: 30px; height: 1px; background: var(--border-strong); }
.oh-diamond { color: var(--gold); font-size: 0.5rem; }
.order-close {
    position: absolute; right: 0.75rem; top: 0.75rem; background: none; border: none;
    color: var(--text-muted); cursor: pointer; font-size: 1rem; transition: var(--transition);
}
.order-close:hover { color: var(--red); }

.order-body { flex: 1; overflow-y: auto; padding: 1.5rem; }
.order-section { margin-bottom: 1.25rem; }
.order-section-title {
    font-family: var(--font-display); font-size: 0.75rem; color: var(--gold);
    letter-spacing: 0.08em; margin-bottom: 0.5rem; padding-bottom: 0.25rem;
    border-bottom: 1px solid var(--border);
}
.order-field { margin-bottom: 0.75rem; }
.of-label { font-size: 0.55rem; color: var(--text-muted); letter-spacing: 0.08em; margin-bottom: 0.2rem; text-transform: uppercase; }
.of-value { font-size: 0.8rem; color: var(--text-primary); line-height: 1.5; }

.order-verdict-banner {
    padding: 0.75rem 1rem; border-radius: var(--radius);
    font-family: var(--font-display); font-size: 0.85rem; font-weight: 600;
    text-align: center; letter-spacing: 0.04em;
}
.order-verdict-banner.verified { background: var(--green-glow); color: var(--green); border: 1px solid rgba(76,175,138,0.3); }
.order-verdict-banner.hallucinated { background: var(--red-glow); color: var(--red); border: 1px solid rgba(232,119,119,0.3); }
.order-verdict-banner.skipped { background: var(--amber-glow); color: var(--amber); border: 1px solid rgba(232,184,119,0.3); }
.order-verdict-banner.no-match { background: rgba(170,170,170,0.1); color: #aaa; border: 1px solid rgba(170,170,170,0.2); }

.order-quote-block {
    background: var(--bg-deep); padding: 0.75rem 1rem; border-radius: var(--radius);
    border-left: 3px solid var(--gold); font-style: italic;
    font-family: var(--font-legal); font-size: 0.85rem; line-height: 1.7;
}

.order-source-paragraph {
    background: var(--bg-deep); padding: 0.75rem 1rem; border-radius: var(--radius);
    border-left: 3px solid var(--purple); font-size: 0.75rem; line-height: 1.6;
    max-height: 200px; overflow-y: auto; color: var(--text-secondary);
}

.order-quote-section {
    background: var(--purple-glow); padding: 0.75rem; border-radius: var(--radius);
    border: 1px solid rgba(184,119,232,0.15);
}
.order-quote-section .order-section-title { color: var(--purple); border-bottom-color: rgba(184,119,232,0.2); }

.order-footer {
    padding: 1rem; border-top: 1px solid var(--border);
    text-align: center;
}
.stamp-circle {
    display: inline-flex; align-items: center; justify-content: center;
    width: 60px; height: 60px; border: 2px solid var(--gold);
    border-radius: 50%; margin-bottom: 0.5rem;
}
.stamp-circle span {
    font-size: 0.5rem; font-weight: 700; letter-spacing: 0.08em;
    color: var(--gold); transform: rotate(-15deg);
}
.order-disclaimer { font-size: 0.6rem; color: var(--text-muted); font-style: italic; }

/* ===== SEARCH TAB ===== */
.search-panel-body { padding: 1.5rem; }
.search-description { font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem; }
.search-input-group { display: flex; gap: 0.5rem; margin-bottom: 0.75rem; }
.search-input-wrap {
    flex: 1; position: relative; display: flex; align-items: center;
}
.search-icon { position: absolute; left: 0.75rem; color: var(--text-muted); font-size: 0.8rem; }
.manual-search-input {
    width: 100%; padding: 0.6rem 0.75rem 0.6rem 2.25rem;
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius); color: var(--text-primary);
    font-size: 0.8rem; font-family: var(--font-body); transition: var(--transition);
}
.manual-search-input:focus { outline: none; border-color: var(--gold); box-shadow: 0 0 0 2px var(--gold-glow); }
.search-btn {
    padding: 0.6rem 1rem; background: linear-gradient(135deg, var(--gold-dark), var(--gold));
    border: none; border-radius: var(--radius); color: var(--bg-deep);
    font-weight: 700; font-size: 0.75rem; cursor: pointer; transition: var(--transition);
    font-family: var(--font-body); white-space: nowrap;
}
.search-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(201,168,76,0.3); }
.search-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.search-examples { display: flex; gap: 0.35rem; flex-wrap: wrap; align-items: center; margin-bottom: 1rem; }
.ex-label { font-size: 0.65rem; color: var(--text-muted); }
.ex-chip {
    font-size: 0.6rem; padding: 0.2rem 0.5rem; background: var(--bg-elevated);
    border: 1px solid var(--border); border-radius: var(--radius); color: var(--text-secondary);
    cursor: pointer; transition: var(--transition); font-family: var(--font-body);
}
.ex-chip:hover { border-color: var(--gold); color: var(--gold); }

.search-results-area { min-height: 200px; }
.search-idle, .search-loading, .search-error { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 200px; gap: 0.75rem; }
.search-error { color: var(--red); }
.search-loading p, .search-idle p { font-size: 0.8rem; color: var(--text-secondary); }

.search-result-card {
    padding: 1rem; background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius-lg); border-left: 4px solid var(--border);
}
.search-result-card.verified { border-left-color: var(--green); }
.search-result-card.hallucinated { border-left-color: var(--red); }
.search-result-card.skipped { border-left-color: var(--amber); }
.src-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
.src-citation { font-family: var(--font-legal); font-size: 1rem; font-weight: 600; }
.src-verdict { font-size: 0.75rem; font-weight: 700; }
.src-details { display: flex; flex-direction: column; gap: 0.4rem; }
.src-field { display: flex; gap: 0.5rem; font-size: 0.75rem; }
.src-label { font-weight: 600; color: var(--text-muted); min-width: 60px; font-size: 0.6rem; letter-spacing: 0.06em; }

/* ===== BULK TAB ===== */
.bulk-panel-body { padding: 1.5rem; }
.bulk-dropzone {
    padding: 2rem; text-align: center; border: 2px dashed var(--border-strong);
    border-radius: var(--radius-lg); cursor: pointer; transition: var(--transition);
}
.bulk-dropzone:hover, .bulk-dropzone.drag-over { border-color: var(--gold); background: var(--gold-glow); }
.bulk-dropzone i { font-size: 2rem; color: var(--gold); margin-bottom: 0.5rem; }
.bulk-dropzone h3 { font-family: var(--font-display); font-size: 0.85rem; color: var(--gold); margin-bottom: 0.25rem; }
.bulk-dropzone p { font-size: 0.75rem; color: var(--text-secondary); }

.bulk-file-list { margin-top: 0.75rem; display: flex; flex-direction: column; gap: 0.35rem; }
.bulk-file-item {
    display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0.75rem;
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius);
}
.bfi-name { flex: 1; font-size: 0.75rem; word-break: break-all; }
.bfi-size { font-size: 0.6rem; color: var(--text-muted); font-family: var(--font-mono); }
.bfi-remove { background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 0.7rem; }
.bfi-remove:hover { color: var(--red); }

.bulk-progress { margin-top: 1rem; }
.bulk-progress-bar { height: 6px; background: var(--bg-deep); border-radius: 3px; overflow: hidden; }
.bulk-progress-fill { height: 100%; background: var(--gold); border-radius: 3px; transition: width 0.3s ease; width: 0; }
.bulk-progress-text { font-size: 0.7rem; color: var(--text-secondary); margin-top: 0.3rem; text-align: center; }

.bulk-results-area { margin-top: 1rem; }
.bulk-summary-header {
    display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;
}
.bulk-stat {
    flex: 1; min-width: 80px; text-align: center; padding: 0.75rem;
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius);
}
.bs-num { font-family: var(--font-display); font-size: 1.5rem; font-weight: 700; }
.bs-label { font-size: 0.6rem; color: var(--text-muted); letter-spacing: 0.06em; }
.bulk-stat.upheld .bs-num { color: var(--green); }
.bulk-stat.fabricated .bs-num { color: var(--red); }
.bulk-stat.quote-issues .bs-num { color: var(--purple); }

.bulk-doc-card {
    padding: 0.75rem; background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius); margin-bottom: 0.5rem;
}
.bulk-doc-card.error { border-left: 3px solid var(--red); }
.bdc-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }
.bdc-name { flex: 1; font-size: 0.8rem; font-weight: 600; }
.bdc-badge {
    font-size: 0.6rem; padding: 0.15rem 0.5rem; background: var(--gold-glow);
    border: 1px solid var(--border-strong); border-radius: 3px; color: var(--gold);
}
.bdc-badge.error { background: var(--red-glow); border-color: rgba(232,119,119,0.3); color: var(--red); }
.bdc-stats { display: flex; gap: 0.75rem; font-size: 0.7rem; color: var(--text-secondary); flex-wrap: wrap; }

/* ===== HISTORY TAB ===== */
.history-toolbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.75rem 1rem; border-bottom: 1px solid var(--border);
}
.history-count { font-size: 0.75rem; color: var(--text-secondary); }
.history-list { flex: 1; overflow-y: auto; padding: 0.75rem; }
.history-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 200px; gap: 0.75rem; }
.history-empty p { font-size: 0.8rem; color: var(--text-secondary); }

.history-item {
    padding: 0.75rem; background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius); margin-bottom: 0.4rem; cursor: pointer;
    transition: var(--transition);
}
.history-item:hover { border-color: var(--gold); background: var(--bg-hover); }
.hi-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem; }
.hi-file { font-size: 0.8rem; font-weight: 600; }
.hi-date { font-size: 0.6rem; color: var(--text-muted); font-family: var(--font-mono); }
.hi-stats { display: flex; gap: 0.75rem; font-size: 0.65rem; color: var(--text-secondary); flex-wrap: wrap; }

/* ===== FOOTER ===== */
.court-registry {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 1.5rem; border-top: 1px solid var(--border);
    background: var(--bg-panel); font-size: 0.6rem;
}
.registry-badge {
    padding: 0.15rem 0.5rem; background: var(--gold-glow); border: 1px solid var(--border-strong);
    border-radius: 3px; color: var(--gold); letter-spacing: 0.06em; font-weight: 600;
}
.registry-center { color: var(--text-muted); font-family: var(--font-legal); font-style: italic; }
.registry-sep { margin: 0 0.5rem; }
.registry-right { font-family: var(--font-mono); color: var(--text-muted); }

/* ===== TOAST ===== */
.toast-container { position: fixed; top: 80px; right: 1rem; z-index: 2000; display: flex; flex-direction: column; gap: 0.5rem; }
.toast {
    display: flex; align-items: center; gap: 0.5rem; padding: 0.6rem 1rem;
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius); font-size: 0.75rem; color: var(--text-primary);
    box-shadow: var(--shadow); max-width: 400px;
    transform: translateX(120%); transition: transform 0.3s ease, opacity 0.3s ease;
}
.toast.show { transform: translateX(0); }
.toast.hide { opacity: 0; transform: translateX(120%); }
.toast-success { border-left: 3px solid var(--green); }
.toast-success i { color: var(--green); }
.toast-error { border-left: 3px solid var(--red); }
.toast-error i { color: var(--red); }
.toast-warning { border-left: 3px solid var(--amber); }
.toast-warning i { color: var(--amber); }
.toast-info { border-left: 3px solid var(--blue); }
.toast-info i { color: var(--blue); }

/* ===== CHATBOT ===== */
.chat-bubble {
    position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 900;
    display: flex; align-items: center; gap: 0.5rem; padding: 0.6rem 1rem;
    background: linear-gradient(135deg, var(--gold-dark), var(--gold));
    border-radius: 50px; cursor: pointer; box-shadow: 0 4px 20px rgba(201,168,76,0.3);
    transition: var(--transition);
}
.chat-bubble:hover { transform: translateY(-3px); box-shadow: 0 6px 30px rgba(201,168,76,0.4); }
.chat-bubble-icon { font-size: 1.2rem; }
.chat-bubble-label { font-size: 0.7rem; font-weight: 700; color: var(--bg-deep); letter-spacing: 0.06em; }
.chat-notification {
    position: absolute; top: -5px; right: -5px; width: 18px; height: 18px;
    background: var(--red); border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 0.55rem; font-weight: 700; color: white;
}

.chat-pane {
    position: fixed; bottom: 5rem; right: 1.5rem; z-index: 950;
    width: 400px; max-height: 550px;
    background: var(--bg-panel); border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg); box-shadow: var(--shadow-lg);
    display: flex; flex-direction: column; overflow: hidden;
    animation: slideUp 0.3s ease;
}
@keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

.chat-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.75rem; border-bottom: 1px solid var(--border);
    background: var(--bg-elevated);
}
.chat-header-left { display: flex; align-items: center; gap: 0.5rem; }
.chat-avatar { font-size: 1.2rem; }
.chat-title { font-size: 0.75rem; font-weight: 700; color: var(--text-primary); }
.chat-subtitle { font-size: 0.55rem; color: var(--text-muted); }
.chat-header-actions { display: flex; gap: 0.25rem; }
.chat-action-btn {
    background: none; border: 1px solid transparent; color: var(--text-muted);
    cursor: pointer; padding: 0.3rem; border-radius: 4px; font-size: 0.7rem;
    transition: var(--transition);
}
.chat-action-btn:hover { color: var(--text-primary); background: var(--bg-hover); }
.chat-action-btn.active { color: var(--gold); border-color: var(--border-strong); background: var(--gold-glow); }

.chat-context-bar {
    padding: 0.35rem 0.75rem; background: var(--green-glow); border-bottom: 1px solid rgba(76,175,138,0.2);
    font-size: 0.6rem; color: var(--green); display: flex; align-items: center; gap: 0.35rem;
}

.chat-messages { flex: 1; overflow-y: auto; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; min-height: 200px; max-height: 350px; }
.chat-msg { display: flex; gap: 0.5rem; animation: fadeIn 0.3s ease; }
.chat-msg.user { flex-direction: row-reverse; }
.msg-avatar { font-size: 1rem; flex-shrink: 0; }
.msg-bubble {
    max-width: 85%; padding: 0.5rem 0.75rem; border-radius: var(--radius);
    font-size: 0.75rem; line-height: 1.6;
}
.chat-msg.assistant .msg-bubble { background: var(--bg-elevated); border: 1px solid var(--border); }
.chat-msg.user .msg-bubble { background: var(--gold-glow); border: 1px solid var(--border-strong); color: var(--text-primary); }

.typing-indicator { display: flex; gap: 0.3rem; align-items: center; padding: 0.75rem !important; }
.typing-indicator span {
    width: 6px; height: 6px; background: var(--gold); border-radius: 50%;
    animation: typing 1.2s infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing { 0%, 60%, 100% { opacity: 0.3; transform: translateY(0); } 30% { opacity: 1; transform: translateY(-4px); } }

.chat-suggestions { display: flex; gap: 0.3rem; padding: 0.5rem 0.75rem; flex-wrap: wrap; border-top: 1px solid var(--border); }
.chat-chip {
    font-size: 0.6rem; padding: 0.2rem 0.5rem; background: var(--bg-elevated);
    border: 1px solid var(--border); border-radius: 20px; color: var(--text-secondary);
    cursor: pointer; transition: var(--transition); font-family: var(--font-body);
}
.chat-chip:hover { border-color: var(--gold); color: var(--gold); }

.chat-input-area { display: flex; gap: 0.35rem; padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); }
.chat-input {
    flex: 1; padding: 0.4rem 0.6rem; background: var(--bg-elevated);
    border: 1px solid var(--border); border-radius: var(--radius);
    color: var(--text-primary); font-size: 0.75rem; resize: none;
    font-family: var(--font-body); max-height: 100px;
}
.chat-input:focus { outline: none; border-color: var(--gold); }
.chat-send-btn {
    background: var(--gold); border: none; border-radius: var(--radius);
    color: var(--bg-deep); cursor: pointer; padding: 0.4rem 0.6rem;
    font-size: 0.75rem; transition: var(--transition);
}
.chat-send-btn:hover { background: var(--gold-light); }

/* Summary Modal Extras */
.summary-loading { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 200px; gap: 1rem; }
.summary-loading p { font-size: 0.8rem; color: var(--text-secondary); }

/* ===== RESPONSIVE ===== */
@media (max-width: 900px) {
    .courtroom { flex-direction: column; }
    .filing-desk { flex: none; }
    .bench-bar { flex-wrap: wrap; gap: 0.5rem; }
    .bench-center { order: 3; width: 100%; justify-content: center; }
    .bench-right { flex-wrap: wrap; }
    .chat-pane { width: calc(100vw - 2rem); right: 1rem; bottom: 4.5rem; }
}

@media (max-width: 600px) {
    .bench-nav-tabs { flex-wrap: wrap; }
    .verdict-summary { grid-template-columns: repeat(3, 1fr); }
    .search-input-group { flex-direction: column; }
    .order-sheet { max-width: 98%; }
    .j-card-top { flex-direction: column; }
    .j-badges { flex-direction: row; align-items: flex-start; }
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }
```

---

### File: `frontend/js/main.js`

```js
const API_BASE = window.location.origin;

async function checkServerStatus() {
    const dot = document.getElementById('serverStatus');
    const text = document.getElementById('serverStatusText');
    if (!dot || !text) return;
    try {
        const res = await fetch(`${API_BASE}/db-stats`);
        const data = await res.json();
        dot.classList.remove('offline');
        dot.classList.add('online');
        text.textContent = `${data.record_count?.toLocaleString() || 0} records`;
        const dbRecords = document.getElementById('dbRecords');
        if (dbRecords) animateNumber(dbRecords, data.record_count || 0);
    } catch (e) {
        dot.classList.remove('online');
        dot.classList.add('offline');
        text.textContent = 'Offline';
    }
}

function animateNumber(el, target, duration = 1500) {
    const startTime = performance.now();
    function update(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.floor(target * eased).toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function showToast(message, type = 'info') {
    const existing = document.querySelector('.home-toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = `home-toast home-toast-${type}`;
    toast.style.cssText = 'position:fixed;top:85px;right:24px;z-index:2000;display:flex;align-items:center;gap:10px;padding:14px 20px;border-radius:10px;font-size:0.88rem;font-weight:500;animation:fadeInUp 0.4s ease;box-shadow:0 8px 30px rgba(0,0,0,0.3);';
    if (type === 'success') toast.style.cssText += 'background:rgba(76,175,138,0.15);border:1px solid rgba(76,175,138,0.3);color:#4caf8a;';
    else if (type === 'error') toast.style.cssText += 'background:rgba(232,119,119,0.15);border:1px solid rgba(232,119,119,0.3);color:#e87777;';
    else toast.style.cssText += 'background:rgba(119,184,232,0.15);border:1px solid rgba(119,184,232,0.3);color:#77b8e8;';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}

document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus();
    setInterval(checkServerStatus, 30000);
});
```

---

### File: `frontend/js/bail-reckoner.js`

```js
const API_BASE = window.location.origin;

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('bailForm').addEventListener('submit', handleBailSubmit);
});

async function handleBailSubmit(e) {
    e.preventDefault();
    const statute = document.getElementById('statute').value;
    const offenseCategory = document.getElementById('offenseCategory').value;
    const imprisonmentDuration = parseInt(document.getElementById('imprisonmentDuration').value) || 0;
    const riskEscape = document.getElementById('riskEscape').checked;
    const riskInfluence = document.getElementById('riskInfluence').checked;
    const servedHalfTerm = document.getElementById('servedHalfTerm').checked;
    if (!statute || !offenseCategory) return;

    document.getElementById('bailEmpty').style.display = 'none';
    document.getElementById('bailResults').style.display = 'none';
    document.getElementById('bailNoData').style.display = 'none';
    document.getElementById('bailLoading').style.display = 'block';

    try {
        const res = await fetch(`${API_BASE}/reckoner/bail`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ statute, offense_category: offenseCategory, imprisonment_duration_served: imprisonmentDuration, risk_of_escape: riskEscape, risk_of_influence: riskInfluence, served_half_term: servedHalfTerm })
        });
        const data = await res.json();
        document.getElementById('bailLoading').style.display = 'none';

        if (data.status === 'NO_DATA') {
            document.getElementById('bailNoData').style.display = 'block';
            document.getElementById('noDataMessage').textContent = data.message || 'No matching cases found.';
            return;
        }
        if (data.status === 'SUCCESS') renderBailResults(data);
        else document.getElementById('bailEmpty').style.display = 'block';
    } catch (e) {
        document.getElementById('bailLoading').style.display = 'none';
        document.getElementById('bailEmpty').style.display = 'block';
    }
}

function renderBailResults(data) {
    document.getElementById('bailResults').style.display = 'block';
    const insights = data.historical_insights || {};
    const conditions = data.likely_conditions || {};
    const probability = parseFloat(insights.historical_bail_probability || '0');

    animateGauge(probability);
    document.getElementById('casesAnalyzed').textContent = insights.similar_cases_analyzed || '—';
    document.getElementById('riskScore').textContent = insights.average_risk_score !== undefined ? insights.average_risk_score.toFixed(2) : '—';

    const surety = document.getElementById('suretyBond');
    const personal = document.getElementById('personalBond');
    surety.className = conditions.surety_bond_likely ? 'bond-card likely' : 'bond-card unlikely';
    surety.querySelector('.bond-status').textContent = conditions.surety_bond_likely ? 'Likely Required' : 'Unlikely';
    personal.className = conditions.personal_bond_likely ? 'bond-card likely' : 'bond-card unlikely';
    personal.querySelector('.bond-status').textContent = conditions.personal_bond_likely ? 'Likely Required' : 'Unlikely';

    document.getElementById('strategyText').textContent = data.legal_strategy_note || 'Standard bail arguments apply.';
    const warning = document.getElementById('statutoryWarning');
    if (data.legal_strategy_note && data.legal_strategy_note.includes('⚠️')) {
        warning.style.display = 'flex';
        document.getElementById('statutoryWarningText').textContent = data.legal_strategy_note;
    } else {
        warning.style.display = 'none';
    }
}

function animateGauge(percentage) {
    const arc = document.getElementById('gaugeArc');
    const valueEl = document.getElementById('gaugeValue');
    const totalLength = 251.2;
    const startTime = performance.now();

    function animate(now) {
        const progress = Math.min((now - startTime) / 1500, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = percentage * eased;
        arc.setAttribute('stroke-dashoffset', totalLength - (totalLength * (current / 100)));
        valueEl.textContent = `${Math.round(current)}%`;
        valueEl.style.color = current > 70 ? '#4caf8a' : current > 40 ? '#e8b877' : '#e87777';
        if (progress < 1) requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
}
```

---

### File: `frontend/js/citation-auditor.js`

```js
// ==========================================
// CITATION-AUDITOR.JS
// ==========================================

let selectedFiles = [];
let auditResults = null;
let chatHistory = [];

document.addEventListener('DOMContentLoaded', () => {
    initUploadZone();
    initButtons();
    initChat();
    initFilters();
});

// ==========================================
// UPLOAD ZONE
// ==========================================
function initUploadZone() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');
    
    zone.addEventListener('click', () => input.click());
    
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });
    
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });
    
    input.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

function handleFiles(fileList) {
    for (const file of fileList) {
        if (file.type !== 'application/pdf') {
            showToast(`"${file.name}" is not a PDF file`, 'error');
            continue;
        }
        if (selectedFiles.some(f => f.name === file.name)) {
            showToast(`"${file.name}" already added`, 'info');
            continue;
        }
        selectedFiles.push(file);
    }
    renderFileList();
    updateButtons();
}

function renderFileList() {
    const list = document.getElementById('fileList');
    list.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <div class="file-item-info">
                <i class="fas fa-file-pdf"></i>
                <span class="file-item-name">${file.name}</span>
            </div>
            <span class="file-item-size">${formatFileSize(file.size)}</span>
            <button class="file-item-remove" onclick="removeFile(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        list.appendChild(item);
    });
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
    updateButtons();
}

function updateButtons() {
    document.getElementById('btnAudit').disabled = selectedFiles.length === 0;
    document.getElementById('btnClear').disabled = selectedFiles.length === 0;
}

// ==========================================
// BUTTONS
// ==========================================
function initButtons() {
    document.getElementById('btnAudit').addEventListener('click', startAudit);
    document.getElementById('btnClear').addEventListener('click', clearAll);
    document.getElementById('btnQuickVerify').addEventListener('click', quickVerify);
    
    // Quick verify on Enter key
    document.getElementById('quickCitation').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') quickVerify();
    });
}

function clearAll() {
    selectedFiles = [];
    auditResults = null;
    renderFileList();
    updateButtons();
    
    document.getElementById('summaryBanner').style.display = 'none';
    document.getElementById('aiSummary').style.display = 'none';
    document.getElementById('filterTabs').style.display = 'none';
    document.getElementById('progressSection').style.display = 'none';
    
    document.getElementById('resultsContainer').innerHTML = `
        <div class="empty-state">
            <div class="empty-icon"><i class="fas fa-balance-scale"></i></div>
            <h3>No Audit Results Yet</h3>
            <p>Upload a legal document to begin citation verification</p>
        </div>
    `;
    
    showToast('Cleared all data', 'info');
}

// ==========================================
// QUICK VERIFY
// ==========================================
async function quickVerify() {
    const input = document.getElementById('quickCitation');
    const resultDiv = document.getElementById('quickResult');
    const citation = input.value.trim();
    
    if (!citation) {
        showToast('Please enter a citation to verify', 'error');
        return;
    }
    
    resultDiv.innerHTML = `
        <div class="file-item" style="border-color: var(--gold-glow); margin-top: 8px;">
            <div class="file-item-info">
                <i class="fas fa-spinner fa-spin" style="color: var(--gold);"></i>
                <span>Verifying citation...</span>
            </div>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_BASE}/verify-citation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ citation })
        });
        
        const data = await response.json();
        const v = data.verification || {};
        const status = v.status || 'Unknown';
        
        let statusClass = 'skipped';
        let icon = 'question-circle';
        
        if (status.includes('VERIFIED')) {
            statusClass = 'verified';
            icon = 'check-circle';
        } else if (status.includes('HALLUCINATION')) {
            statusClass = 'fabricated';
            icon = 'exclamation-triangle';
        }
        
        resultDiv.innerHTML = `
            <div class="result-card" style="margin-top: 8px;">
                <div class="result-card-header">
                    <div class="result-card-left">
                        <div class="result-status-icon ${statusClass}">
                            <i class="fas fa-${icon}"></i>
                        </div>
                        <div>
                            <div class="result-citation-name">${citation}</div>
                            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">
                                ${v.matched_name || v.message || 'No match found'}
                            </div>
                        </div>
                    </div>
                </div>
                ${v.reason ? `
                <div style="padding: 12px 18px; border-top: 1px solid var(--border-color);">
                    <div class="detail-label">Reason</div>
                    <div class="detail-value">${v.reason}</div>
                </div>` : ''}
                ${v.confidence !== undefined ? `
                <div style="padding: 0 18px 12px;">
                    <div class="detail-label">Confidence</div>
                    <div class="detail-value">${v.confidence}%</div>
                </div>` : ''}
            </div>
        `;
    } catch (e) {
        resultDiv.innerHTML = `
            <div class="file-item" style="border-color: var(--red-border); margin-top: 8px;">
                <div class="file-item-info">
                    <i class="fas fa-exclamation-circle" style="color: var(--red);"></i>
                    <span style="color: var(--red);">Error: ${e.message}</span>
                </div>
            </div>
        `;
    }
}

// ==========================================
// MAIN AUDIT
// ==========================================
async function startAudit() {
    if (selectedFiles.length === 0) return;
    
    const progressSection = document.getElementById('progressSection');
    const btnAudit = document.getElementById('btnAudit');
    const btnClear = document.getElementById('btnClear');
    
    progressSection.style.display = 'block';
    btnAudit.disabled = true;
    btnClear.disabled = true;
    
    // Reset progress steps
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        step.classList.remove('active', 'done');
    }
    
    updateProgress(10, 'Uploading documents...', 1);
    
    try {
        const formData = new FormData();
        const isMultiple = selectedFiles.length > 1;
        
        if (isMultiple) {
            selectedFiles.forEach(f => formData.append('files', f));
        } else {
            formData.append('file', selectedFiles[0]);
        }
        
        updateProgress(20, 'Extracting text and identifying citations...', 2);
        
        const endpoint = isMultiple ? '/audit-multiple' : '/audit-document';
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        updateProgress(60, 'Cross-referencing database...', 3);
        
        const data = await response.json();
        
        updateProgress(80, 'Verifying quotations...', 4);
        
        // Process results
        let results;
        if (isMultiple) {
            results = [];
            (data.documents || []).forEach(doc => {
                (doc.results || []).forEach(r => {
                    r._filename = doc.filename;
                    results.push(r);
                });
            });
            auditResults = {
                results,
                total: data.total_sc_citations + data.total_hc_citations,
                sc_count: data.total_sc_citations,
                hc_count: data.total_hc_citations
            };
        } else {
            results = data.results || [];
            auditResults = {
                results,
                total: data.total_citations_found,
                sc_count: data.supreme_court_count,
                hc_count: data.high_court_count
            };
        }
        
        updateProgress(90, 'Generating report...', 5);
        
        renderResults(auditResults);
        
        // Get AI summary
        await generateSummary(auditResults);
        
        updateProgress(100, 'Audit complete!', 5);
        
        // Mark all steps done
        for (let i = 1; i <= 5; i++) {
            document.getElementById(`step${i}`).classList.remove('active');
            document.getElementById(`step${i}`).classList.add('done');
        }
        
        showToast('Audit completed successfully!', 'success');
        
        setTimeout(() => {
            progressSection.style.display = 'none';
        }, 2000);
        
    } catch (e) {
        showToast(`Audit failed: ${e.message}`, 'error');
        progressSection.style.display = 'none';
    }
    
    btnAudit.disabled = false;
    btnClear.disabled = false;
}

function updateProgress(percent, text, activeStep) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = text;
    
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        if (i < activeStep) {
            step.classList.remove('active');
            step.classList.add('done');
        } else if (i === activeStep) {
            step.classList.add('active');
            step.classList.remove('done');
        }
    }
}

// ==========================================
// RENDER RESULTS
// ==========================================
function renderResults(data) {
    const results = data.results || [];
    
    // Count categories
    let verified = 0, fabricated = 0, skipped = 0;
    results.forEach(r => {
        const status = (r.verification || {}).status || '';
        if (status.includes('VERIFIED') && !status.includes('HC-')) verified++;
        else if (status.includes('HALLUCINATION')) fabricated++;
        else skipped++;
    });
    
    // Update summary banner
    const banner = document.getElementById('summaryBanner');
    banner.style.display = 'flex';
    document.getElementById('totalCount').textContent = results.length;
    document.getElementById('verifiedCount').textContent = verified;
    document.getElementById('fabricatedCount').textContent = fabricated;
    document.getElementById('skippedCount').textContent = skipped;
    
    // Risk badge
    const riskBadge = document.getElementById('riskBadge');
    const riskLevel = document.getElementById('riskLevel');
    if (fabricated > 2) {
        riskBadge.className = 'risk-badge risk-high';
        riskLevel.textContent = 'High Risk';
    } else if (fabricated > 0) {
        riskBadge.className = 'risk-badge risk-medium';
        riskLevel.textContent = 'Medium Risk';
    } else {
        riskBadge.className = 'risk-badge risk-low';
        riskLevel.textContent = 'Low Risk';
    }
    
    // Show filter tabs
    document.getElementById('filterTabs').style.display = 'flex';
    
    // Render cards
    renderResultCards(results, 'all');
}

function renderResultCards(results, filter) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';
    
    const filtered = results.filter(r => {
        if (filter === 'all') return true;
        const status = (r.verification || {}).status || '';
        if (filter === 'verified') return status.includes('VERIFIED') && !status.includes('HC-');
        if (filter === 'fabricated') return status.includes('HALLUCINATION');
        if (filter === 'skipped') return !status.includes('VERIFIED') && !status.includes('HALLUCINATION');
        return true;
    });
    
    if (filtered.length === 0) {
        container.innerHTML = `
            <div class="empty-state" style="padding: 3rem;">
                <p style="color: var(--text-muted);">No results match this filter.</p>
            </div>
        `;
        return;
    }
    
    filtered.forEach((r, idx) => {
        const v = r.verification || {};
        const q = r.quote_verification || {};
        const status = v.status || 'Unknown';
        
        let statusClass = 'skipped';
        let icon = 'question-circle';
        
        if (status.includes('VERIFIED') && !status.includes('HC-')) {
            statusClass = 'verified';
            icon = 'check-circle';
        } else if (status.includes('HALLUCINATION')) {
            statusClass = 'fabricated';
            icon = 'exclamation-triangle';
        }
        
        const courtType = (r.court_type || '').includes('High Court') ? 'hc' : 'sc';
        const courtLabel = courtType === 'hc' ? 'HC' : 'SC';
        
        // Quote verification status
        let qStatusClass = 'skipped';
        let qLabel = q.status || 'N/A';
        if (qLabel.includes('VERIFIED')) qStatusClass = 'verified';
        else if (qLabel.includes('CONTRADICTED')) qStatusClass = 'contradicted';
        else if (qLabel.includes('PARTIAL')) qStatusClass = 'partial';
        
        const card = document.createElement('div');
        card.className = `result-card`;
        card.dataset.status = statusClass;
        card.innerHTML = `
            <div class="result-card-header" onclick="toggleCard(this)">
                <div class="result-card-left">
                    <div class="result-status-icon ${statusClass}">
                        <i class="fas fa-${icon}"></i>
                    </div>
                    <span class="result-citation-name">${r.target_citation || 'Unknown'}</span>
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <span class="result-court-badge ${courtType}">${courtLabel}</span>
                    <button class="result-expand-btn">
                        <i class="fas fa-chevron-down"></i>
                    </button>
                </div>
            </div>
            <div class="result-card-details" id="details-${idx}">
                <div class="detail-section">
                    <div class="detail-label">Verification Status</div>
                    <div class="detail-value">${status}</div>
                </div>
                ${v.matched_name ? `
                <div class="detail-section">
                    <div class="detail-label">Matched Case</div>
                    <div class="detail-value">${v.matched_name}</div>
                </div>` : ''}
                ${v.matched_citation ? `
                <div class="detail-section">
                    <div class="detail-label">Database Citation</div>
                    <div class="detail-value mono">${v.matched_citation}</div>
                </div>` : ''}
                ${v.reason ? `
                <div class="detail-section">
                    <div class="detail-label">Matching Reason</div>
                    <div class="detail-value">${v.reason}</div>
                </div>` : ''}
                ${v.message ? `
                <div class="detail-section">
                    <div class="detail-label">Note</div>
                    <div class="detail-value">${v.message}</div>
                </div>` : ''}
                ${v.confidence !== undefined ? `
                <div class="detail-section">
                    <div class="detail-label">Confidence</div>
                    <div class="detail-value">${v.confidence}%</div>
                </div>` : ''}
                <div class="detail-section">
                    <div class="detail-label">Quote Verification</div>
                    <div>
                        <span class="quote-status ${qStatusClass}">${qLabel}</span>
                    </div>
                    ${q.reason ? `<div class="detail-value" style="margin-top:8px;">${q.reason}</div>` : ''}
                    ${q.verdict ? `<div class="detail-value" style="margin-top:4px; font-weight:600;">Verdict: ${q.verdict}</div>` : ''}
                </div>
                ${r._filename ? `
                <div class="detail-section">
                    <div class="detail-label">Source File</div>
                    <div class="detail-value mono">${r._filename}</div>
                </div>` : ''}
            </div>
        `;
        container.appendChild(card);
    });
}

function toggleCard(header) {
    const card = header.closest('.result-card');
    const details = card.querySelector('.result-card-details');
    const icon = header.querySelector('.result-expand-btn i');
    
    details.classList.toggle('open');
    icon.classList.toggle('fa-chevron-down');
    icon.classList.toggle('fa-chevron-up');
}

// ==========================================
// FILTERS
// ==========================================
function initFilters() {
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('filter-tab') || e.target.closest('.filter-tab')) {
            const tab = e.target.classList.contains('filter-tab') ? e.target : e.target.closest('.filter-tab');
            
            document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const filter = tab.dataset.filter;
            if (auditResults) {
                renderResultCards(auditResults.results, filter);
            }
        }
    });
}

// ==========================================
// AI SUMMARY
// ==========================================
async function generateSummary(data) {
    try {
        const response = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: data.results,
                total: data.total,
                sc_count: data.sc_count,
                hc_count: data.hc_count
            })
        });
        
        const summaryData = await response.json();
        
        const aiSummary = document.getElementById('aiSummary');
        aiSummary.style.display = 'block';
        document.getElementById('aiSummaryText').textContent = summaryData.summary;
        
    } catch (e) {
        console.error('Summary generation failed:', e);
    }
}

// ==========================================
// CHAT WIDGET
// ==========================================
function initChat() {
    const toggle = document.getElementById('chatToggle');
    const panel = document.getElementById('chatPanel');
    const close = document.getElementById('chatClose');
    const input = document.getElementById('chatInput');
    const send = document.getElementById('chatSend');
    
    toggle.addEventListener('click', () => {
        panel.classList.toggle('open');
    });
    
    close.addEventListener('click', () => {
        panel.classList.remove('open');
    });
    
    send.addEventListener('click', sendChatMessage);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const messages = document.getElementById('chatMessages');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user-message';
    userMsg.innerHTML = `
        <div class="message-avatar"><i class="fas fa-user"></i></div>
        <div class="message-content"><p>${escapeHtml(message)}</p></div>
    `;
    messages.appendChild(userMsg);
    
    input.value = '';
    messages.scrollTop = messages.scrollHeight;
    
    // Show typing indicator
    const typing = document.createElement('div');
    typing.className = 'chat-message bot-message';
    typing.id = 'typing-indicator';
    typing.innerHTML = `
        <div class="message-avatar"><i class="fas fa-robot"></i></div>
        <div class="message-content"><p><i class="fas fa-circle" style="font-size:0.4rem;animation:pulse-dot 1s infinite;"></i> 
        <i class="fas fa-circle" style="font-size:0.4rem;animation:pulse-dot 1s infinite 0.2s;"></i> 
        <i class="fas fa-circle" style="font-size:0.4rem;animation:pulse-dot 1s infinite 0.4s;"></i></p></div>
    `;
    messages.appendChild(typing);
    messages.scrollTop = messages.scrollHeight;
    
    // Build audit context
    let auditContext = '';
    if (auditResults) {
        const verified = auditResults.results.filter(r => (r.verification?.status || '').includes('VERIFIED')).length;
        const fabricated = auditResults.results.filter(r => (r.verification?.status || '').includes('HALLUCINATION')).length;
        auditContext = `Current audit: ${auditResults.total} citations found, ${verified} verified, ${fabricated} fabricated.`;
    }
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                history: chatHistory,
                audit_context: auditContext || null
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        document.getElementById('typing-indicator')?.remove();
        
        // Add bot response
        const botMsg = document.createElement('div');
        botMsg.className = 'chat-message bot-message';
        botMsg.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content"><p>${formatChatResponse(data.reply)}</p></div>
        `;
        messages.appendChild(botMsg);
        messages.scrollTop = messages.scrollHeight;
        
        // Update history
        chatHistory.push({ role: 'user', content: message });
        chatHistory.push({ role: 'assistant', content: data.reply });
        
        // Keep history manageable
        if (chatHistory.length > 20) {
            chatHistory = chatHistory.slice(-20);
        }
        
    } catch (e) {
        document.getElementById('typing-indicator')?.remove();
        
        const errorMsg = document.createElement('div');
        errorMsg.className = 'chat-message bot-message';
        errorMsg.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content"><p style="color:var(--red);">Sorry, I encountered an error. Please try again.</p></div>
        `;
        messages.appendChild(errorMsg);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatChatResponse(text) {
    // Basic formatting
    let formatted = escapeHtml(text);
    
    // Bold text between **
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Bullet points
    formatted = formatted.replace(/^[-•]\s+(.+)$/gm, '<br>• $1');
    
    // Newlines to br
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}
```

---

### File: `frontend/templates/auditor.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚖️ Legal Citation Auditor — Supreme Court of India</title>
    <meta name="description" content="AI-powered legal citation verification system with RAG-based quote verification">
    <link rel="stylesheet" href="/static/styles.css">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,500&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500&family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
</head>
<body>

    <!-- ========== OATH SCREEN (Boot) ========== -->
    <div id="oath-screen" class="oath-screen">
        <div class="oath-content">
            <div class="oath-emblem">
                <div class="emblem-outer-ring"></div>
                <div class="emblem-inner-ring"></div>
                <div class="emblem-icon">⚖️</div>
            </div>
            <div class="oath-heading">SUPREME COURT OF INDIA</div>
            <div class="oath-subheading">Citation Integrity & Quote Verification System v2.1</div>
            <div class="oath-divider">
                <span class="divider-wing left"></span>
                <span class="divider-diamond">◆</span>
                <span class="divider-wing right"></span>
            </div>
            <div class="oath-lines">
                <div class="oath-line" style="--delay: 0.2s">
                    <span class="oath-bullet">§</span>
                    <span>Initializing Judicial AI Engine...</span>
                    <span class="oath-check">✓</span>
                </div>
                <div class="oath-line" style="--delay: 0.6s">
                    <span class="oath-bullet">§</span>
                    <span>Loading Supreme Court Case Archive...</span>
                    <span class="oath-check">✓</span>
                </div>
                <div class="oath-line" style="--delay: 1.0s">
                    <span class="oath-bullet">§</span>
                    <span>Connecting to Groq LLM Inference...</span>
                    <span class="oath-check">✓</span>
                </div>
                <div class="oath-line" style="--delay: 1.4s">
                    <span class="oath-bullet">§</span>
                    <span>Loading RAG Embedding Model (MiniLM-L6)...</span>
                    <span class="oath-check">✓</span>
                </div>
                <div class="oath-line" style="--delay: 1.8s">
                    <span class="oath-bullet">§</span>
                    <span>Engaging Hallucination Detection Protocol...</span>
                    <span class="oath-check">✓</span>
                </div>
                <div class="oath-line" style="--delay: 2.2s">
                    <span class="oath-bullet">§</span>
                    <span>Activating Quote Verification Engine...</span>
                    <span class="oath-check">✓</span>
                </div>
                <div class="oath-line" style="--delay: 2.6s">
                    <span class="oath-bullet">§</span>
                    <span>Upholding the Rule of Law...</span>
                    <span class="oath-check">✓</span>
                </div>
            </div>
            <div class="oath-progress">
                <div class="oath-progress-track">
                    <div class="oath-progress-fill"></div>
                </div>
            </div>
            <div class="oath-footer-text">
                "Yato Dharmastato Jayah" — Where there is Dharma, there is Victory
            </div>
        </div>
    </div>

    <!-- ========== MAIN APPLICATION ========== -->
    <div id="app" class="app hidden">

        <!-- ===== HEADER / BENCH BAR ===== -->
        <header class="bench-bar">
            <div class="bench-left">
                <div class="bench-emblem">
                    <a href="/app" style="text-decoration:none;" title="Back to Home">
                        <div class="mini-scales">⚖️</div>
                    </a>
                </div>
                <div class="bench-title-block">
                    <h1 class="bench-title">CITATION AUDITOR <span class="version-badge">v2.1</span></h1>
                    <p class="bench-subtitle">Supreme Court of India • AI Verification Chamber + RAG</p>
                </div>
            </div>
            <div class="bench-center">
                <div class="bench-status-row">
                    <div class="bench-indicator online" id="ind-api">
                        <div class="indicator-lamp"></div>
                        <span>API</span>
                    </div>
                    <div class="bench-indicator online" id="ind-llm">
                        <div class="indicator-lamp"></div>
                        <span>LLM</span>
                    </div>
                    <div class="bench-indicator online" id="ind-db">
                        <div class="indicator-lamp"></div>
                        <span>DATABASE</span>
                    </div>
                    <div class="bench-indicator online" id="ind-rag">
                        <div class="indicator-lamp"></div>
                        <span>RAG</span>
                    </div>
                </div>
            </div>
            <div class="bench-right">
                <div class="bench-nav-tabs">
                    <button class="nav-tab" onclick="window.location.href='/app'" title="Back to Home">
                        <i class="fas fa-home"></i> HOME
                    </button>
                    <button class="nav-tab active" id="nav-audit" onclick="switchTab('audit')">
                        <i class="fas fa-gavel"></i> AUDIT
                    </button>
                    <button class="nav-tab" id="nav-search" onclick="switchTab('search')">
                        <i class="fas fa-search"></i> SEARCH
                    </button>
                    <button class="nav-tab" id="nav-bulk" onclick="switchTab('bulk')">
                        <i class="fas fa-layer-group"></i> BULK
                    </button>
                    <button class="nav-tab" id="nav-history" onclick="switchTab('history')">
                        <i class="fas fa-history"></i> HISTORY
                    </button>
                </div>
                <div class="bench-clock" id="bench-clock"></div>
                <div class="bench-session">
                    <span class="session-label">SESSION</span>
                    <span class="session-id" id="session-id">—</span>
                </div>
            </div>
        </header>

        <!-- ===== TAB CONTENT: AUDIT ===== -->
        <main class="courtroom" id="tab-audit">
            <section class="court-panel filing-desk">
                <div class="panel-title-bar">
                    <div class="title-bar-ornament left">━━◆</div>
                    <h2>FILING DESK</h2>
                    <div class="title-bar-ornament right">◆━━</div>
                </div>
                <div class="seal-dropzone" id="seal-dropzone">
                    <div class="seal-visual">
                        <div class="wax-seal">
                            <div class="seal-ring r1"></div>
                            <div class="seal-ring r2"></div>
                            <div class="seal-ring r3"></div>
                            <div class="seal-center"><i class="fas fa-file-pdf"></i></div>
                        </div>
                    </div>
                    <h3>FILE YOUR DOCUMENT</h3>
                    <p>Drop a legal PDF or click to select</p>
                    <div class="format-tags">
                        <span class="f-tag">PDF ONLY</span>
                        <span class="f-tag">≤ 50 MB</span>
                    </div>
                    <input type="file" id="file-input" accept=".pdf" hidden>
                </div>
                <div class="filed-doc hidden" id="filed-doc">
                    <div class="filed-header">
                        <div class="filed-icon"><i class="fas fa-file-contract"></i></div>
                        <div class="filed-info">
                            <span class="filed-name" id="filed-name">—</span>
                            <span class="filed-size" id="filed-size">—</span>
                        </div>
                        <button class="filed-remove" id="filed-remove" title="Remove"><i class="fas fa-times"></i></button>
                    </div>
                    <div class="filed-stamp"><span>DOCUMENT FILED</span><i class="fas fa-stamp"></i></div>
                </div>
                <button class="gavel-btn" id="gavel-btn" disabled>
                    <div class="gavel-bg"></div>
                    <div class="gavel-content">
                        <span class="gavel-icon">🔨</span>
                        <span class="gavel-text">ORDER! COMMENCE AUDIT</span>
                    </div>
                    <div class="gavel-shine"></div>
                </button>
                <div class="verdict-summary" id="verdict-summary">
                    <div class="verdict-card total"><div class="vc-icon"><i class="fas fa-scroll"></i></div><div class="vc-number" id="vc-total">—</div><div class="vc-label">CITATIONS</div></div>
                    <div class="verdict-card upheld"><div class="vc-icon"><i class="fas fa-gavel"></i></div><div class="vc-number" id="vc-upheld">—</div><div class="vc-label">UPHELD</div></div>
                    <div class="verdict-card overruled"><div class="vc-icon"><i class="fas fa-ban"></i></div><div class="vc-number" id="vc-overruled">—</div><div class="vc-label">FABRICATED</div></div>
                    <div class="verdict-card skipped"><div class="vc-icon"><i class="fas fa-university"></i></div><div class="vc-number" id="vc-skipped">—</div><div class="vc-label">HIGH COURT</div></div>
                    <div class="verdict-card unheard"><div class="vc-icon"><i class="fas fa-question-circle"></i></div><div class="vc-number" id="vc-unheard">—</div><div class="vc-label">UNVERIFIED</div></div>
                </div>
                <div class="verdict-summary quote-summary hidden" id="quote-summary">
                    <div class="quote-summary-title"><i class="fas fa-quote-left"></i> QUOTE VERIFICATION (RAG)</div>
                    <div class="verdict-card quote-verified"><div class="vc-icon"><i class="fas fa-check-double"></i></div><div class="vc-number" id="vc-quote-verified">—</div><div class="vc-label">QUOTES OK</div></div>
                    <div class="verdict-card quote-contradicted"><div class="vc-icon"><i class="fas fa-exclamation-triangle"></i></div><div class="vc-number" id="vc-quote-contradicted">—</div><div class="vc-label">CONTRADICTED</div></div>
                    <div class="verdict-card quote-unsupported"><div class="vc-icon"><i class="fas fa-question"></i></div><div class="vc-number" id="vc-quote-unsupported">—</div><div class="vc-label">UNSUPPORTED</div></div>
                    <div class="verdict-card quote-fabricated"><div class="vc-icon"><i class="fas fa-ghost"></i></div><div class="vc-number" id="vc-quote-fabricated">—</div><div class="vc-label">FABRICATED QUOTE</div></div>
                </div>
                <div class="court-breakdown hidden" id="court-breakdown">
                    <div class="cb-title"><i class="fas fa-balance-scale-left"></i><span>COURT CLASSIFICATION</span></div>
                    <div class="cb-row"><div class="cb-bar-label">Supreme Court</div><div class="cb-bar-track"><div class="cb-bar-fill sc-fill" id="cb-sc-fill"></div></div><div class="cb-bar-count" id="cb-sc-count">0</div></div>
                    <div class="cb-row"><div class="cb-bar-label">High Court</div><div class="cb-bar-track"><div class="cb-bar-fill hc-fill" id="cb-hc-fill"></div></div><div class="cb-bar-count" id="cb-hc-count">0</div></div>
                </div>
                <div class="post-audit-actions hidden" id="post-audit-actions">
                    <button class="action-btn export-pdf-btn" onclick="exportPDF()"><i class="fas fa-file-pdf"></i> Export PDF</button>
                    <button class="action-btn export-csv-btn" onclick="exportCSV()"><i class="fas fa-file-csv"></i> Export CSV</button>
                    <button class="action-btn summary-btn" onclick="generateSummary()"><i class="fas fa-file-alt"></i> AI Summary</button>
                </div>
            </section>

            <section class="court-panel judgment-chamber">
                <div class="panel-title-bar">
                    <div class="title-bar-ornament left">━━◆</div>
                    <h2>JUDGMENT CHAMBER</h2>
                    <div class="title-bar-ornament right">◆━━</div>
                </div>
                <div class="chamber-idle" id="chamber-idle">
                    <div class="idle-scales-container"><div class="scales-beam"><div class="scales-pivot">⚖</div></div></div>
                    <h3>THE COURT IS IN SESSION</h3>
                    <p>File a document to begin judicial review of cited authorities</p>
                    <p class="idle-rag-note"><i class="fas fa-brain"></i> RAG-powered quote verification enabled</p>
                </div>
                <div class="chamber-deliberation hidden" id="chamber-deliberation">
                    <div class="deliberation-visual"><div class="quill-animation"><div class="quill-body"><i class="fas fa-feather-alt"></i></div><div class="ink-drops"><span class="ink-drop" style="--d:0"></span><span class="ink-drop" style="--d:1"></span><span class="ink-drop" style="--d:2"></span></div></div></div>
                    <h3 class="delib-title" id="delib-title">COURT IN DELIBERATION</h3>
                    <p class="delib-sub" id="delib-sub">The bench is reviewing your document...</p>
                    <div class="delib-steps">
                        <div class="d-step active" id="ds-1"><div class="d-step-marker">I</div><span>Reading the Document</span></div>
                        <div class="d-step" id="ds-2"><div class="d-step-marker">II</div><span>Identifying &amp; Classifying Authorities</span></div>
                        <div class="d-step" id="ds-3"><div class="d-step-marker">III</div><span>Filtering High Court Citations</span></div>
                        <div class="d-step" id="ds-4"><div class="d-step-marker">IV</div><span>Verifying SC Cases Against Registry</span></div>
                        <div class="d-step" id="ds-5"><div class="d-step-marker">V</div><span>RAG Quote Verification</span></div>
                        <div class="d-step" id="ds-6"><div class="d-step-marker">VI</div><span>Pronouncing Judgment</span></div>
                    </div>
                </div>
                <div class="chamber-results hidden" id="chamber-results">
                    <div class="judgment-filters">
                        <button class="jf-tab active" data-filter="all">ALL MATTERS <span class="jf-count" id="jf-all">0</span></button>
                        <button class="jf-tab" data-filter="verified">UPHELD <span class="jf-count" id="jf-upheld">0</span></button>
                        <button class="jf-tab" data-filter="hallucinated">FABRICATED <span class="jf-count" id="jf-fabricated">0</span></button>
                        <button class="jf-tab" data-filter="skipped">HIGH COURT <span class="jf-count" id="jf-skipped">0</span></button>
                        <button class="jf-tab" data-filter="no-match">UNVERIFIED <span class="jf-count" id="jf-unverified">0</span></button>
                        <button class="jf-tab" data-filter="quote-issue">QUOTE ISSUES <span class="jf-count" id="jf-quote-issues">0</span></button>
                    </div>
                    <div class="judgment-roll" id="judgment-roll"></div>
                </div>
            </section>
        </main>

        <!-- ===== SEARCH TAB ===== -->
        <main class="courtroom search-tab hidden" id="tab-search">
            <section class="court-panel search-panel" style="flex:1;">
                <div class="panel-title-bar"><div class="title-bar-ornament left">━━◆</div><h2>MANUAL CITATION SEARCH</h2><div class="title-bar-ornament right">◆━━</div></div>
                <div class="search-panel-body">
                    <p class="search-description">Enter any case name or citation to instantly verify it against the Supreme Court archive.</p>
                    <div class="search-input-group">
                        <div class="search-input-wrap"><i class="fas fa-search search-icon"></i><input type="text" id="manual-search-input" placeholder="e.g. State of Bihar v. Ram Kumar Singh..." class="manual-search-input" onkeypress="if(event.key==='Enter') runManualSearch()"></div>
                        <button class="search-btn" id="search-btn" onclick="runManualSearch()"><i class="fas fa-gavel"></i> VERIFY</button>
                    </div>
                    <div class="search-examples">
                        <span class="ex-label">Examples:</span>
                        <button class="ex-chip" onclick="fillSearch('Vashist Narayan Kumar v. State of Bihar')">Vashist Narayan Kumar v. State of Bihar</button>
                        <button class="ex-chip" onclick="fillSearch('Divya vs. Union of India')">Divya vs. Union of India</button>
                        <button class="ex-chip" onclick="fillSearch('Maneka Gandhi v. Union of India')">Maneka Gandhi v. Union of India</button>
                    </div>
                    <div class="search-results-area" id="search-results-area">
                        <div class="search-idle"><i class="fas fa-balance-scale" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i><p>Enter a citation above to verify it</p></div>
                    </div>
                </div>
            </section>
        </main>

        <!-- ===== BULK TAB ===== -->
        <main class="courtroom bulk-tab hidden" id="tab-bulk">
            <section class="court-panel bulk-panel" style="flex:1;">
                <div class="panel-title-bar"><div class="title-bar-ornament left">━━◆</div><h2>BULK DOCUMENT AUDIT</h2><div class="title-bar-ornament right">◆━━</div></div>
                <div class="bulk-panel-body">
                    <p class="search-description">Upload multiple PDF documents at once for a combined citation audit.</p>
                    <div class="bulk-dropzone" id="bulk-dropzone" onclick="document.getElementById('bulk-file-input').click()">
                        <i class="fas fa-layer-group"></i><h3>DROP MULTIPLE PDFS HERE</h3><p>Or click to select files</p>
                        <input type="file" id="bulk-file-input" accept=".pdf" multiple hidden>
                    </div>
                    <div class="bulk-file-list" id="bulk-file-list"></div>
                    <button class="gavel-btn" id="bulk-audit-btn" disabled style="margin-top:1rem;" onclick="runBulkAudit()">
                        <div class="gavel-bg"></div><div class="gavel-content"><span class="gavel-icon">📋</span><span class="gavel-text">AUDIT ALL DOCUMENTS</span></div><div class="gavel-shine"></div>
                    </button>
                    <div class="bulk-progress hidden" id="bulk-progress"><div class="bulk-progress-bar"><div class="bulk-progress-fill" id="bulk-progress-fill"></div></div><div class="bulk-progress-text" id="bulk-progress-text">Processing...</div></div>
                    <div class="bulk-results-area" id="bulk-results-area"></div>
                </div>
            </section>
        </main>

        <!-- ===== HISTORY TAB ===== -->
        <main class="courtroom history-tab hidden" id="tab-history">
            <section class="court-panel history-panel" style="flex:1;">
                <div class="panel-title-bar"><div class="title-bar-ornament left">━━◆</div><h2>AUDIT HISTORY</h2><div class="title-bar-ornament right">◆━━</div></div>
                <div class="history-toolbar"><span id="history-count" class="history-count">0 records</span><button class="action-btn" onclick="clearHistory()" style="background:rgba(220,50,50,0.15);border-color:rgba(220,50,50,0.3);color:#e87777;"><i class="fas fa-trash"></i> Clear All</button></div>
                <div class="history-list" id="history-list"><div class="history-empty"><i class="fas fa-history" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i><p>No audit history yet.</p></div></div>
            </section>
        </main>

        <!-- ===== FOOTER ===== -->
        <footer class="court-registry">
            <div class="registry-left"><span class="registry-badge">POWERED BY GROQ × LLAMA 3.3 70B × MiniLM-L6 RAG</span></div>
            <div class="registry-center"><span>यतो धर्मस्ततो जयः</span><span class="registry-sep">•</span><span>Where there is Righteousness, there is Victory</span></div>
            <div class="registry-right"><span id="registry-records">—</span></div>
        </footer>
    </div>

    <!-- ========== MODALS ========== -->
    <div class="order-overlay hidden" id="order-overlay">
        <div class="order-sheet" id="order-sheet">
            <div class="order-header"><div class="order-header-ornament"><span class="oh-line"></span><span class="oh-diamond">◆</span><span class="oh-line"></span></div><h3>JUDGMENT &amp; ORDER</h3><div class="order-header-ornament"><span class="oh-line"></span><span class="oh-diamond">◆</span><span class="oh-line"></span></div><button class="order-close" id="order-close"><i class="fas fa-times"></i></button></div>
            <div class="order-body" id="order-body"></div>
            <div class="order-footer"><div class="order-seal"><div class="stamp-circle"><span>VERIFIED</span></div></div><p class="order-disclaimer">This AI-generated audit is for informational purposes only and does not constitute legal advice.</p></div>
        </div>
    </div>

    <div class="order-overlay hidden" id="summary-overlay">
        <div class="order-sheet">
            <div class="order-header"><div class="order-header-ornament"><span class="oh-line"></span><span class="oh-diamond">◆</span><span class="oh-line"></span></div><h3>AI AUDIT SUMMARY</h3><div class="order-header-ornament"><span class="oh-line"></span><span class="oh-diamond">◆</span><span class="oh-line"></span></div><button class="order-close" onclick="document.getElementById('summary-overlay').classList.add('hidden')"><i class="fas fa-times"></i></button></div>
            <div class="order-body" id="summary-body"><div class="summary-loading"><i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Generating professional summary...</p></div></div>
            <div class="order-footer"><button class="action-btn export-pdf-btn" onclick="exportSummaryPDF()" style="margin:0 auto;"><i class="fas fa-download"></i> Download Summary</button></div>
        </div>
    </div>

    <div id="toast-container" class="toast-container"></div>

    <!-- ========== LEXAI CHATBOT ========== -->
    <div class="chat-bubble" id="chat-bubble" onclick="toggleChat()">
        <div class="chat-bubble-icon">⚖️</div>
        <div class="chat-bubble-label">LexAI</div>
        <div class="chat-notification" id="chat-notification" style="display:none;">1</div>
    </div>

    <div class="chat-pane hidden" id="chat-pane">
        <div class="chat-header">
            <div class="chat-header-left"><div class="chat-avatar">⚖️</div><div><div class="chat-title">LexAI Legal Assistant</div><div class="chat-subtitle">Powered by Groq × Llama 3.3 + RAG</div></div></div>
            <div class="chat-header-actions">
                <button title="Use audit context" onclick="toggleAuditContext()" id="ctx-btn" class="chat-action-btn"><i class="fas fa-link"></i></button>
                <button title="Clear chat" onclick="clearChat()" class="chat-action-btn"><i class="fas fa-trash"></i></button>
                <button onclick="toggleChat()" class="chat-action-btn"><i class="fas fa-times"></i></button>
            </div>
        </div>
        <div class="chat-context-bar hidden" id="chat-context-bar"><i class="fas fa-link"></i> Audit context attached</div>
        <div class="chat-messages" id="chat-messages">
            <div class="chat-msg assistant"><div class="msg-avatar">⚖️</div><div class="msg-bubble"><p>Namaste! I'm <strong>LexAI</strong>, your AI legal assistant.</p><p style="margin-top:0.5rem;">I can help with case law, audit analysis, and Indian constitutional law.</p></div></div>
        </div>
        <div class="chat-suggestions" id="chat-suggestions">
            <button class="chat-chip" onclick="sendSuggestion('What is a hallucinated citation?')">What is a hallucinated citation?</button>
            <button class="chat-chip" onclick="sendSuggestion('Explain Article 21')">Article 21</button>
            <button class="chat-chip" onclick="sendSuggestion('How does RAG quote verification work?')">How does RAG work?</button>
        </div>
        <div class="chat-input-area">
            <textarea id="chat-input" class="chat-input" placeholder="Ask any legal question..." rows="1" onkeypress="handleChatKey(event)" oninput="autoResizeChat(this)"></textarea>
            <button class="chat-send-btn" id="chat-send-btn" onclick="sendChatMessage()"><i class="fas fa-paper-plane"></i></button>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    <script src="/static/app.js"></script>
</body>
</html>
```

---

### File: `frontend/templates/bail-reckoner.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚖️ Bail Reckoner — AI Risk Assessment Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,500&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500&family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    <style>
        :root {
            --bg-deepest: #0a0a0f;
            --bg-dark: #0d0d14;
            --bg-panel: #111118;
            --bg-card: #16161f;
            --bg-elevated: #1a1a25;
            --bg-hover: #1e1e2a;
            --gold: #c9a84c;
            --gold-light: #e8d48b;
            --gold-dark: #8b6914;
            --gold-glow: rgba(201, 168, 76, 0.15);
            --gold-glow-strong: rgba(201, 168, 76, 0.35);
            --crimson: #8b2035;
            --crimson-light: #c4445a;
            --emerald: #1a7a4c;
            --emerald-light: #4caf8a;
            --emerald-glow: rgba(26, 122, 76, 0.2);
            --amber: #b8860b;
            --amber-light: #e8b877;
            --text-primary: #e8e4dc;
            --text-secondary: #9a9488;
            --text-muted: #5a5650;
            --border-subtle: rgba(201, 168, 76, 0.08);
            --border-gold: rgba(201, 168, 76, 0.2);
            --border-strong: rgba(201, 168, 76, 0.35);
            --font-display: 'Playfair Display', serif;
            --font-body: 'Cormorant Garamond', serif;
            --font-ui: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background: var(--bg-deepest);
            color: var(--text-primary);
            font-family: var(--font-body);
            min-height: 100vh;
        }

        .hidden { display: none !important; }

        .bg-grid {
            position: fixed; inset: 0; z-index: 0;
            background-image:
                linear-gradient(rgba(201,168,76,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(201,168,76,0.02) 1px, transparent 1px);
            background-size: 80px 80px;
            pointer-events: none;
        }

        /* ── Header ── */
        .bail-header {
            position: sticky; top: 0; z-index: 100;
            background: rgba(10,10,15,0.92);
            backdrop-filter: blur(30px);
            border-bottom: 1px solid var(--border-subtle);
            padding: 0 2rem;
        }

        .bail-header-inner {
            max-width: 1200px; margin: 0 auto;
            height: 65px; display: flex;
            align-items: center; justify-content: space-between;
        }

        .bail-header-left {
            display: flex; align-items: center; gap: 1rem;
        }

        .back-btn {
            display: flex; align-items: center; gap: 0.5rem;
            text-decoration: none; color: var(--text-secondary);
            font-family: var(--font-ui); font-size: 0.75rem;
            letter-spacing: 1px; text-transform: uppercase;
            padding: 0.4rem 0.8rem; border: 1px solid var(--border-subtle);
            border-radius: 4px; transition: all 0.3s ease;
        }

        .back-btn:hover {
            color: var(--gold); border-color: var(--border-gold);
        }

        .bail-header-title {
            font-family: var(--font-display);
            font-size: 1.1rem; font-weight: 700;
            color: var(--emerald-light);
            letter-spacing: 2px; text-transform: uppercase;
        }

        .bail-header-badge {
            font-family: var(--font-mono); font-size: 0.55rem;
            color: var(--text-muted); letter-spacing: 1px;
            padding: 0.2rem 0.5rem; border: 1px solid var(--border-subtle);
            border-radius: 3px;
        }

        .bail-header-right {
            display: flex; align-items: center; gap: 0.75rem;
        }

        .db-status {
            display: flex; align-items: center; gap: 0.4rem;
            font-family: var(--font-mono); font-size: 0.55rem;
            letter-spacing: 0.5px;
        }

        .db-status-dot {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--text-muted);
            transition: background 0.3s ease;
        }

        .db-status-dot.online { background: var(--emerald-light); box-shadow: 0 0 6px var(--emerald-glow); }
        .db-status-dot.offline { background: var(--crimson-light); }

        .db-status-text { color: var(--text-muted); }

        /* ── Main Content ── */
        .bail-main {
            position: relative; z-index: 1;
            max-width: 1200px; margin: 0 auto;
            padding: 2.5rem 2rem;
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 2.5rem;
        }

        /* ── Form Panel ── */
        .form-panel {
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            border-radius: 12px; padding: 2rem;
        }

        .panel-header {
            display: flex; align-items: center; gap: 0.75rem;
            margin-bottom: 2rem; padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .panel-header-icon {
            width: 40px; height: 40px;
            border: 1.5px solid var(--emerald-light);
            border-radius: 50%; display: flex;
            align-items: center; justify-content: center;
            font-size: 1.1rem;
        }

        .panel-header h2 {
            font-family: var(--font-display);
            font-size: 1.2rem; font-weight: 700;
            color: var(--text-primary); letter-spacing: 1px;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-label {
            display: block; font-family: var(--font-ui);
            font-size: 0.7rem; font-weight: 600;
            letter-spacing: 1.5px; text-transform: uppercase;
            color: var(--text-secondary); margin-bottom: 0.5rem;
        }

        .form-hint {
            font-family: var(--font-ui); font-size: 0.6rem;
            color: var(--text-muted); margin-top: 0.3rem;
            font-style: italic;
        }

        .form-input, .form-select {
            width: 100%; padding: 0.75rem 1rem;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 6px; color: var(--text-primary);
            font-family: var(--font-ui); font-size: 0.85rem;
            transition: all 0.3s ease;
            outline: none;
        }

        .form-input:focus, .form-select:focus {
            border-color: var(--emerald-light);
            box-shadow: 0 0 20px rgba(76,175,138,0.1);
        }

        .form-input.input-error, .form-select.input-error {
            border-color: var(--crimson-light);
            box-shadow: 0 0 15px rgba(139,32,53,0.15);
        }

        .form-select option {
            background: var(--bg-card); color: var(--text-primary);
        }

        .form-checkbox-group {
            display: flex; gap: 1.5rem; flex-wrap: wrap;
        }

        .form-checkbox {
            display: flex; align-items: center; gap: 0.5rem;
            cursor: pointer;
        }

        .form-checkbox input[type="checkbox"] {
            appearance: none; width: 18px; height: 18px;
            border: 1.5px solid var(--border-gold);
            border-radius: 3px; background: var(--bg-card);
            cursor: pointer; position: relative;
            transition: all 0.3s ease;
            flex-shrink: 0;
        }

        .form-checkbox input[type="checkbox"]:checked {
            background: var(--emerald-light);
            border-color: var(--emerald-light);
        }

        .form-checkbox input[type="checkbox"]:checked::after {
            content: '✓'; position: absolute;
            inset: 0; display: flex; align-items: center;
            justify-content: center; color: var(--bg-deepest);
            font-size: 0.7rem; font-weight: 700;
        }

        .form-checkbox-label {
            font-family: var(--font-ui); font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .submit-btn {
            width: 100%; padding: 1rem;
            background: linear-gradient(135deg, var(--emerald), var(--emerald-light));
            border: none; border-radius: 8px;
            color: white; font-family: var(--font-display);
            font-size: 1rem; font-weight: 700;
            letter-spacing: 2px; text-transform: uppercase;
            cursor: pointer; position: relative;
            overflow: hidden; transition: all 0.4s ease;
            margin-top: 1rem;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(76,175,138,0.3);
        }

        .submit-btn:disabled {
            opacity: 0.5; cursor: not-allowed;
            transform: none; box-shadow: none;
        }

        .submit-btn .btn-shine {
            position: absolute; top: 0; left: -100%;
            width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        }

        .submit-btn:hover .btn-shine {
            animation: btnShine 0.8s ease forwards;
        }

        @keyframes btnShine {
            to { left: 100%; }
        }

        /* ── Results Panel ── */
        .results-panel {
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            border-radius: 12px; padding: 2rem;
            display: flex; flex-direction: column;
        }

        .results-idle {
            flex: 1; display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            text-align: center; gap: 1rem; opacity: 0.5;
        }

        .results-idle i {
            font-size: 3rem; color: var(--emerald-light);
        }

        .results-idle p {
            font-family: var(--font-ui); font-size: 0.85rem;
            color: var(--text-muted); max-width: 300px;
        }

        /* ── Loading ── */
        .results-loading {
            flex: 1; display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            gap: 1.5rem;
        }

        .loading-scales {
            font-size: 3rem;
            animation: loadingBounce 1.2s ease-in-out infinite;
        }

        @keyframes loadingBounce {
            0%, 100% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.1) rotate(5deg); }
        }

        .loading-text {
            font-family: var(--font-display);
            font-size: 1rem; color: var(--emerald-light);
            letter-spacing: 2px;
        }

        .loading-sub {
            font-family: var(--font-ui);
            font-size: 0.75rem; color: var(--text-muted);
        }

        .loading-bar {
            width: 200px; height: 3px;
            background: var(--bg-card); border-radius: 2px;
            overflow: hidden;
        }

        .loading-bar-fill {
            height: 100%; width: 30%;
            background: var(--emerald-light);
            border-radius: 2px;
            animation: loadingSlide 1.5s ease-in-out infinite;
        }

        @keyframes loadingSlide {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(700%); }
        }

        /* ── Result Card ── */
        .result-card {
            animation: resultSlideIn 0.5s ease forwards;
        }

        @keyframes resultSlideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-verdict {
            text-align: center; padding: 1.5rem;
            border-radius: 8px; margin-bottom: 1.5rem;
        }

        .result-verdict.verdict-high {
            background: rgba(76,175,138,0.08);
            border: 1px solid rgba(76,175,138,0.2);
        }
        .result-verdict.verdict-high .result-verdict-text { color: var(--emerald-light); }

        .result-verdict.verdict-medium {
            background: rgba(184,134,11,0.08);
            border: 1px solid rgba(184,134,11,0.2);
        }
        .result-verdict.verdict-medium .result-verdict-text { color: var(--amber-light); }

        .result-verdict.verdict-low {
            background: rgba(139,32,53,0.08);
            border: 1px solid rgba(139,32,53,0.2);
        }
        .result-verdict.verdict-low .result-verdict-text { color: var(--crimson-light); }

        .result-verdict.no-data {
            background: rgba(232,184,119,0.08);
            border: 1px solid rgba(232,184,119,0.2);
        }
        .result-verdict.no-data .result-verdict-text { color: var(--amber-light); }

        .result-verdict.error {
            background: rgba(139,32,53,0.08);
            border: 1px solid rgba(139,32,53,0.2);
        }
        .result-verdict.error .result-verdict-text { color: var(--crimson-light); }

        .result-verdict-icon { font-size: 2rem; margin-bottom: 0.5rem; }

        .result-verdict-text {
            font-family: var(--font-display);
            font-size: 1.1rem; font-weight: 700;
        }

        .result-section { margin-bottom: 1.5rem; }

        .result-section-title {
            font-family: var(--font-mono);
            font-size: 0.6rem; letter-spacing: 2px;
            text-transform: uppercase; color: var(--text-muted);
            margin-bottom: 0.75rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .result-row {
            display: flex; justify-content: space-between;
            align-items: center; padding: 0.5rem 0;
            border-bottom: 1px solid rgba(201,168,76,0.04);
        }

        .result-row:last-child { border-bottom: none; }

        .result-row-label {
            font-family: var(--font-ui);
            font-size: 0.75rem; color: var(--text-secondary);
        }

        .result-row-value {
            font-family: var(--font-mono);
            font-size: 0.8rem; font-weight: 600;
            color: var(--text-primary);
        }

        .result-row-value.highlight { color: var(--emerald-light); font-size: 1rem; }
        .result-row-value.warning { color: var(--amber-light); }
        .result-row-value.danger { color: var(--crimson-light); }

        /* ── Probability Gauge ── */
        .probability-gauge {
            position: relative; width: 160px; height: 160px;
            margin: 1.5rem auto;
        }

        .gauge-ring {
            width: 100%; height: 100%;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            transition: all 0.8s ease;
        }

        .gauge-inner {
            width: 120px; height: 120px;
            border-radius: 50%;
            background: var(--bg-panel);
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
        }

        .gauge-pct {
            font-family: var(--font-display);
            font-size: 2rem; font-weight: 800;
        }

        .gauge-pct.gauge-high { color: var(--emerald-light); }
        .gauge-pct.gauge-medium { color: var(--amber-light); }
        .gauge-pct.gauge-low { color: var(--crimson-light); }

        .gauge-label {
            font-family: var(--font-mono);
            font-size: 0.45rem; letter-spacing: 2px;
            color: var(--text-muted); text-transform: uppercase;
        }

        /* ── Risk Meter ── */
        .risk-meter { margin: 1rem 0; }

        .risk-meter-bar {
            width: 100%; height: 6px;
            background: var(--bg-card);
            border-radius: 3px; overflow: hidden;
            margin-top: 0.5rem;
        }

        .risk-meter-fill {
            height: 100%; border-radius: 3px;
            transition: width 0.8s ease, background 0.8s ease;
        }

        .risk-meter-labels {
            display: flex; justify-content: space-between;
            margin-top: 0.3rem;
        }

        .risk-meter-labels span {
            font-family: var(--font-mono);
            font-size: 0.5rem; color: var(--text-muted);
            letter-spacing: 1px; text-transform: uppercase;
        }

        /* ── Strategy Note ── */
        .strategy-note {
            padding: 1rem; border-radius: 6px;
            background: rgba(201,168,76,0.05);
            border: 1px solid rgba(201,168,76,0.15);
            margin-top: 1rem;
        }

        .strategy-note.statutory-warning {
            background: rgba(76,175,138,0.08);
            border: 1px solid rgba(76,175,138,0.25);
        }

        .strategy-note-title {
            font-family: var(--font-ui);
            font-size: 0.65rem; font-weight: 600;
            color: var(--gold); letter-spacing: 1.5px;
            text-transform: uppercase; margin-bottom: 0.5rem;
        }

        .strategy-note.statutory-warning .strategy-note-title {
            color: var(--emerald-light);
        }

        .strategy-note-text {
            font-family: var(--font-body);
            font-size: 0.95rem; color: var(--text-secondary);
            line-height: 1.6; font-style: italic;
        }

        /* ── Bond Badges ── */
        .bond-badges {
            display: flex; gap: 0.75rem; margin-top: 0.5rem;
            flex-wrap: wrap;
        }

        .bond-badge {
            padding: 0.4rem 0.8rem;
            border-radius: 4px; font-family: var(--font-mono);
            font-size: 0.6rem; letter-spacing: 1px;
            text-transform: uppercase;
        }

        .bond-badge.likely {
            background: rgba(76,175,138,0.1);
            border: 1px solid rgba(76,175,138,0.3);
            color: var(--emerald-light);
        }

        .bond-badge.unlikely {
            background: rgba(154,148,136,0.05);
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
        }

        /* ── Risk Flags ── */
        .risk-flags {
            display: flex; gap: 0.5rem; flex-wrap: wrap;
            margin-top: 0.5rem;
        }

        .risk-flag {
            padding: 0.3rem 0.6rem;
            border-radius: 4px; font-family: var(--font-mono);
            font-size: 0.55rem; letter-spacing: 0.5px;
        }

        .risk-flag.active {
            background: rgba(139,32,53,0.12);
            border: 1px solid rgba(196,68,90,0.3);
            color: var(--crimson-light);
        }

        .risk-flag.inactive {
            background: rgba(76,175,138,0.06);
            border: 1px solid rgba(76,175,138,0.15);
            color: var(--emerald-light);
        }

        /* ── Error Display ── */
        .error-detail {
            padding: 0.75rem 1rem;
            background: rgba(139,32,53,0.08);
            border: 1px solid rgba(139,32,53,0.2);
            border-radius: 6px; margin-top: 1rem;
        }

        .error-detail-text {
            font-family: var(--font-mono);
            font-size: 0.7rem; color: var(--crimson-light);
            word-break: break-word;
        }

        /* ── Penalty Info ── */
        .penalty-badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 3px;
            font-family: var(--font-mono);
            font-size: 0.6rem;
            letter-spacing: 0.5px;
        }

        .penalty-badge.imprisonment {
            background: rgba(139,32,53,0.1);
            border: 1px solid rgba(139,32,53,0.25);
            color: var(--crimson-light);
        }

        .penalty-badge.fine {
            background: rgba(184,134,11,0.1);
            border: 1px solid rgba(184,134,11,0.25);
            color: var(--amber-light);
        }

        .penalty-badge.both {
            background: rgba(201,168,76,0.1);
            border: 1px solid rgba(201,168,76,0.25);
            color: var(--gold-light);
        }

        /* ── Timestamp ── */
        .result-timestamp {
            font-family: var(--font-mono);
            font-size: 0.55rem; color: var(--text-muted);
            text-align: right; margin-top: 1.5rem;
            letter-spacing: 0.5px;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border-subtle);
        }

        /* ── Footer ── */
        .bail-footer {
            position: relative; z-index: 1;
            text-align: center; padding: 2rem;
            border-top: 1px solid var(--border-subtle);
        }

        .bail-footer-text {
            font-family: var(--font-ui);
            font-size: 0.6rem; color: var(--text-muted);
            letter-spacing: 1.5px;
        }

        @media (max-width: 800px) {
            .bail-main { grid-template-columns: 1fr; }
            .bail-header-badge { display: none; }
            .bail-header-inner { height: 55px; }
            .bail-main { padding: 1.5rem 1rem; }
            .form-panel, .results-panel { padding: 1.5rem; }
            .db-status { display: none; }
        }
    </style>
</head>
<body>

    <div class="bg-grid"></div>

    <!-- ══════════════════════════════════════════════
         HEADER
    ══════════════════════════════════════════════ -->
    <header class="bail-header">
        <div class="bail-header-inner">
            <div class="bail-header-left">
                <a href="/" class="back-btn">
                    <i class="fas fa-arrow-left"></i> Platform Home
                </a>
                <span class="bail-header-title">⚖️ Bail Reckoner</span>
            </div>
            <div class="bail-header-right">
                <div class="db-status">
                    <div class="db-status-dot" id="db-dot"></div>
                    <span class="db-status-text" id="db-text">Checking DB...</span>
                </div>
                <span class="bail-header-badge">AI Risk Assessment Engine v1.0</span>
            </div>
        </div>
    </header>

    <!-- ══════════════════════════════════════════════
         MAIN
    ══════════════════════════════════════════════ -->
    <main class="bail-main">

        <!-- ─── Form Panel ─── -->
        <div class="form-panel">
            <div class="panel-header">
                <div class="panel-header-icon">📋</div>
                <h2>Case Parameters</h2>
            </div>

            <form id="bail-form" onsubmit="event.preventDefault(); calculateBail();">

                <!-- Statute -->
                <div class="form-group">
                    <label class="form-label">Statute / Act</label>
                    <select class="form-select" id="bail-statute" required>
                        <option value="">— Select Statute —</option>
                        <option value="IPC">Indian Penal Code (IPC)</option>
                        <option value="NDPS">NDPS Act</option>
                        <option value="PMLA">Prevention of Money Laundering Act (PMLA)</option>
                        <option value="UAPA">UAPA</option>
                        <option value="Arms Act">Arms Act</option>
                        <option value="IT Act">Information Technology Act</option>
                        <option value="CrPC">Code of Criminal Procedure (CrPC)</option>
                        <option value="SCST Act">SC/ST Prevention of Atrocities Act</option>
                        <option value="POCSO">POCSO Act</option>
                        <option value="Other">Other</option>
                    </select>
                    <div class="form-hint">Must match database records exactly (e.g. "SCST Act", "PMLA", "NDPS")</div>
                </div>

                <!-- Offense Category -->
                <div class="form-group">
                    <label class="form-label">Offense Category</label>
                    <select class="form-select" id="bail-category" required>
                        <option value="">— Select Category —</option>
                        <option value="Crimes Against Children">Crimes Against Children</option>
                        <option value="Crimes Against Foreigners">Crimes Against Foreigners</option>
                        <option value="Crimes Against SCs and STs">Crimes Against SCs and STs</option>
                        <option value="Offences Against the State">Offences Against the State</option>
                        <option value="Cyber Crime">Cyber Crime</option>
                        <option value="Economic Offense">Economic Offense</option>
                        <option value="Violent Crime">Violent Crime</option>
                        <option value="Drug Offense">Drug Offense</option>
                        <option value="White Collar Crime">White Collar Crime</option>
                        <option value="Property Crime">Property Crime</option>
                        <option value="Terrorism">Terrorism Related</option>
                        <option value="Other">Other</option>
                    </select>
                    <div class="form-hint">Categories sourced from historical bail database records</div>
                </div>

                <!-- Duration Served -->
                <div class="form-group">
                    <label class="form-label">Imprisonment Duration Served (Days)</label>
                    <input type="number" class="form-input" id="bail-duration" 
                           placeholder="e.g. 180" min="0" max="10000" required>
                    <div class="form-hint">Number of days the accused has been incarcerated</div>
                </div>

                <!-- Risk Factors -->
                <div class="form-group">
                    <label class="form-label">Risk Factors</label>
                    <div class="form-checkbox-group">
                        <label class="form-checkbox">
                            <input type="checkbox" id="bail-flight">
                            <span class="form-checkbox-label">Risk of Flight / Escape</span>
                        </label>
                        <label class="form-checkbox">
                            <input type="checkbox" id="bail-influence">
                            <span class="form-checkbox-label">Risk of Influencing Witnesses</span>
                        </label>
                    </div>
                </div>

                <!-- Statutory Considerations -->
                <div class="form-group">
                    <label class="form-label">Statutory Considerations</label>
                    <div class="form-checkbox-group">
                        <label class="form-checkbox">
                            <input type="checkbox" id="bail-halfterm">
                            <span class="form-checkbox-label">Has served half of maximum term (§436A CrPC)</span>
                        </label>
                    </div>
                </div>

                <button type="submit" class="submit-btn" id="bail-submit">
                    <span class="btn-shine"></span>
                    <i class="fas fa-gavel"></i> &nbsp; CALCULATE BAIL ELIGIBILITY
                </button>
            </form>
        </div>

        <!-- ─── Results Panel ─── -->
        <div class="results-panel" id="results-panel">
            <div class="panel-header">
                <div class="panel-header-icon">📊</div>
                <h2>Assessment Report</h2>
            </div>

            <div class="results-idle" id="results-idle">
                <i class="fas fa-balance-scale"></i>
                <p>Fill in the case parameters and submit to generate an AI-powered bail eligibility assessment.</p>
            </div>

            <div class="results-loading hidden" id="results-loading">
                <div class="loading-scales">⚖️</div>
                <div class="loading-text">ANALYZING CASE</div>
                <div class="loading-sub">Cross-referencing historical bail data...</div>
                <div class="loading-bar"><div class="loading-bar-fill"></div></div>
            </div>

            <div class="result-card hidden" id="result-card">
                <!-- Populated by JS -->
            </div>
        </div>
    </main>

    <!-- ══════════════════════════════════════════════
         FOOTER
    ══════════════════════════════════════════════ -->
    <footer class="bail-footer">
        <div class="bail-footer-text">
            BAIL RECKONER — AI-POWERED RISK ASSESSMENT • POWERED BY GROQ × LLAMA 3.3 70B
        </div>
    </footer>

    <!-- ══════════════════════════════════════════════
         JAVASCRIPT
    ══════════════════════════════════════════════ -->
    <script>
        // ============================================================
        // API BASE — same origin since backend serves this file
        // ============================================================
        const API_BASE = window.location.origin;

        // ============================================================
        // ON LOAD — Check if bail database is loaded
        // ============================================================
        document.addEventListener('DOMContentLoaded', async () => {
            const dot = document.getElementById('db-dot');
            const text = document.getElementById('db-text');
            try {
                const resp = await fetch(`${API_BASE}/api/health`);
                if (resp.ok) {
                    dot.classList.add('online');
                    text.textContent = 'DB Online';
                } else {
                    dot.classList.add('offline');
                    text.textContent = 'DB Error';
                }
            } catch {
                dot.classList.add('offline');
                text.textContent = 'Offline';
            }
        });

        // ============================================================
        // CLEAR VALIDATION STYLES on input change
        // ============================================================
        document.querySelectorAll('.form-select, .form-input').forEach(el => {
            el.addEventListener('change', () => el.classList.remove('input-error'));
            el.addEventListener('input', () => el.classList.remove('input-error'));
        });

        // ============================================================
        // MAIN — Calculate Bail Eligibility
        // ============================================================
        async function calculateBail() {
            // 1. Read form values
            const statuteEl = document.getElementById('bail-statute');
            const categoryEl = document.getElementById('bail-category');
            const durationEl = document.getElementById('bail-duration');

            const statute = statuteEl.value;
            const category = categoryEl.value;
            const durationRaw = durationEl.value;
            const duration = parseInt(durationRaw) || 0;
            const flight = document.getElementById('bail-flight').checked;
            const influence = document.getElementById('bail-influence').checked;
            const halfterm = document.getElementById('bail-halfterm').checked;

            // 2. Client-side validation
            let hasError = false;
            if (!statute) {
                statuteEl.classList.add('input-error');
                hasError = true;
            }
            if (!category) {
                categoryEl.classList.add('input-error');
                hasError = true;
            }
            if (durationRaw === '' || durationRaw === null) {
                durationEl.classList.add('input-error');
                hasError = true;
            }

            if (hasError) {
                showInlineError('Please fill in all required fields before submitting.');
                return;
            }

            // 3. Toggle UI state
            const idleEl = document.getElementById('results-idle');
            const loadingEl = document.getElementById('results-loading');
            const resultEl = document.getElementById('result-card');
            const submitBtn = document.getElementById('bail-submit');

            idleEl.classList.add('hidden');
            resultEl.classList.add('hidden');
            loadingEl.classList.remove('hidden');
            submitBtn.disabled = true;

            // 4. Build request body — matches ReckonerRequest Pydantic model
            //    Fields: statute, offense_category, imprisonment_duration_served,
            //            risk_of_escape, risk_of_influence, served_half_term
            const requestBody = {
                statute: statute,
                offense_category: category,
                imprisonment_duration_served: duration,
                risk_of_escape: flight,
                risk_of_influence: influence,
                served_half_term: halfterm
            };

            try {
                // 5. POST to /reckoner/bail
                const resp = await fetch(`${API_BASE}/reckoner/bail`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });

                // 6. Handle HTTP-level errors (e.g. 500 "Bail database not loaded")
                if (!resp.ok) {
                    let errorMsg = `Server returned ${resp.status}`;
                    try {
                        const errData = await resp.json();
                        // FastAPI HTTPException returns { "detail": "..." }
                        errorMsg = errData.detail || errData.message || errorMsg;
                    } catch (_) {
                        // Response wasn't JSON
                    }
                    throw new Error(errorMsg);
                }

                // 7. Parse JSON response
                const data = await resp.json();

                // 8. Render result
                loadingEl.classList.add('hidden');
                renderResult(data, requestBody);
                resultEl.classList.remove('hidden');

            } catch (err) {
                loadingEl.classList.add('hidden');
                renderError(err.message);
                resultEl.classList.remove('hidden');
            } finally {
                submitBtn.disabled = false;
            }
        }

        // ============================================================
        // RENDER RESULT — Handles SUCCESS and NO_DATA from backend
        //
        // Backend response shapes:
        //
        // NO_DATA:
        //   { status: "NO_DATA", message: "...", recommendation: "..." }
        //
        // SUCCESS:
        //   {
        //     status: "SUCCESS",
        //     inputs_analyzed: { statute, category, time_served },
        //     historical_insights: {
        //       similar_cases_analyzed: int,
        //       historical_bail_probability: "72.5%",   // string with %
        //       average_risk_score: 1.5                  // float
        //     },
        //     likely_conditions: {
        //       surety_bond_likely: true/false,
        //       personal_bond_likely: true/false
        //     },
        //     legal_strategy_note: "string or null"
        //   }
        // ============================================================
        function renderResult(data, requestBody) {
            const el = document.getElementById('result-card');
            const now = new Date().toLocaleString('en-IN', {
                dateStyle: 'medium', timeStyle: 'short', timeZone: 'Asia/Kolkata'
            });

            // ─── NO_DATA response ───────────────────────────────
            if (data.status === 'NO_DATA') {
                el.innerHTML = `
                    <div class="result-verdict no-data">
                        <div class="result-verdict-icon">⚠️</div>
                        <div class="result-verdict-text">${esc(data.message || 'No matching historical data found.')}</div>
                    </div>

                    <div class="result-section">
                        <div class="result-section-title">Submitted Parameters</div>
                        <div class="result-row">
                            <span class="result-row-label">Statute</span>
                            <span class="result-row-value">${esc(requestBody.statute)}</span>
                        </div>
                        <div class="result-row">
                            <span class="result-row-label">Offense Category</span>
                            <span class="result-row-value">${esc(requestBody.offense_category)}</span>
                        </div>
                        <div class="result-row">
                            <span class="result-row-label">Time Served</span>
                            <span class="result-row-value">${requestBody.imprisonment_duration_served} days</span>
                        </div>
                        <div class="result-row">
                            <span class="result-row-label">Flight Risk</span>
                            <span class="result-row-value">${requestBody.risk_of_escape ? 'Yes' : 'No'}</span>
                        </div>
                        <div class="result-row">
                            <span class="result-row-label">Witness Influence Risk</span>
                            <span class="result-row-value">${requestBody.risk_of_influence ? 'Yes' : 'No'}</span>
                        </div>
                        <div class="result-row">
                            <span class="result-row-label">Half-Term Served (§436A)</span>
                            <span class="result-row-value">${requestBody.served_half_term ? 'Yes' : 'No'}</span>
                        </div>
                    </div>

                    <div class="strategy-note">
                        <div class="strategy-note-title"><i class="fas fa-lightbulb"></i> Recommendation</div>
                        <div class="strategy-note-text">${esc(data.recommendation || 'Manual assessment by a legal professional is recommended. Try adjusting the statute or category to match available records.')}</div>
                    </div>

                    <div class="result-timestamp">Report generated: ${now} IST</div>
                `;
                return;
            }

            // ─── SUCCESS response ───────────────────────────────
            const insights = data.historical_insights || {};
            const conditions = data.likely_conditions || {};
            const inputs = data.inputs_analyzed || {};
            const strategyNote = data.legal_strategy_note || '';

            // Parse probability — backend sends "72.5%" as a string
            const probStr = String(insights.historical_bail_probability || '0%');
            const prob = parseFloat(probStr.replace('%', '')) || 0;

            // Risk score — backend sends as float (e.g. 1.5)
            const risk = parseFloat(insights.average_risk_score) || 0;

            // Similar cases — backend sends as int
            const similarCases = parseInt(insights.similar_cases_analyzed) || 0;

            // Bond conditions — backend sends Python booleans serialized as JSON booleans
            const suretyLikely = conditions.surety_bond_likely === true;
            const personalLikely = conditions.personal_bond_likely === true;

            // ── Verdict tier ──
            let verdictClass, verdictIcon, verdictText, gaugeClass, gaugeColor;
            if (prob >= 60) {
                verdictClass = 'verdict-high';
                verdictIcon = '✅';
                verdictText = 'BAIL LIKELY FAVORABLE';
                gaugeClass = 'gauge-high';
                gaugeColor = 'var(--emerald-light)';
            } else if (prob >= 30) {
                verdictClass = 'verdict-medium';
                verdictIcon = '⚠️';
                verdictText = 'BAIL MODERATELY LIKELY';
                gaugeClass = 'gauge-medium';
                gaugeColor = 'var(--amber-light)';
            } else {
                verdictClass = 'verdict-low';
                verdictIcon = '❌';
                verdictText = 'BAIL UNLIKELY — STRONG OPPOSITION EXPECTED';
                gaugeClass = 'gauge-low';
                gaugeColor = 'var(--crimson-light)';
            }

            // ── Risk tier ──
            let riskClass = 'highlight';
            let riskColor = 'var(--emerald-light)';
            let riskLabel = 'Low Risk';
            if (risk > 6) {
                riskClass = 'danger';
                riskColor = 'var(--crimson-light)';
                riskLabel = 'High Risk';
            } else if (risk > 3) {
                riskClass = 'warning';
                riskColor = 'var(--amber-light)';
                riskLabel = 'Moderate Risk';
            }

            // ── Statutory warning detection ──
            const isStatutoryWarning = strategyNote.includes('436A') || strategyNote.includes('statutory');

            // ── Conic gradient degrees ──
            const conicDeg = Math.min(prob * 3.6, 360);

            el.innerHTML = `
                <!-- Verdict Banner -->
                <div class="result-verdict ${verdictClass}">
                    <div class="result-verdict-icon">${verdictIcon}</div>
                    <div class="result-verdict-text">${verdictText}</div>
                </div>

                <!-- Probability Gauge -->
                <div class="probability-gauge">
                    <div class="gauge-ring" style="background: conic-gradient(${gaugeColor} ${conicDeg}deg, var(--bg-card) ${conicDeg}deg);">
                        <div class="gauge-inner">
                            <div class="gauge-pct ${gaugeClass}">${prob.toFixed(1)}%</div>
                            <div class="gauge-label">Bail Probability</div>
                        </div>
                    </div>
                </div>

                <!-- Section I: Historical Insights -->
                <div class="result-section">
                    <div class="result-section-title">I. Historical Insights</div>
                    <div class="result-row">
                        <span class="result-row-label">Similar Cases Analyzed</span>
                        <span class="result-row-value">${similarCases}</span>
                    </div>
                    <div class="result-row">
                        <span class="result-row-label">Historical Bail Probability</span>
                        <span class="result-row-value highlight">${esc(probStr)}</span>
                    </div>
                    <div class="result-row">
                        <span class="result-row-label">Average Risk Score</span>
                        <span class="result-row-value ${riskClass}">${risk.toFixed(2)} / 10</span>
                    </div>

                    <!-- Risk Meter Bar -->
                    <div class="risk-meter">
                        <div class="risk-meter-bar">
                            <div class="risk-meter-fill" style="width: ${Math.min((risk / 10) * 100, 100)}%; background: ${riskColor};"></div>
                        </div>
                        <div class="risk-meter-labels">
                            <span>Low</span>
                            <span>${riskLabel}</span>
                            <span>High</span>
                        </div>
                    </div>
                </div>

                <!-- Section II: Case Parameters -->
                <div class="result-section">
                    <div class="result-section-title">II. Case Parameters Analyzed</div>
                    <div class="result-row">
                        <span class="result-row-label">Statute</span>
                        <span class="result-row-value">${esc(inputs.statute || requestBody.statute)}</span>
                    </div>
                    <div class="result-row">
                        <span class="result-row-label">Category</span>
                        <span class="result-row-value">${esc(inputs.category || requestBody.offense_category)}</span>
                    </div>
                    <div class="result-row">
                        <span class="result-row-label">Time Served</span>
                        <span class="result-row-value">${inputs.time_served != null ? inputs.time_served : requestBody.imprisonment_duration_served} days</span>
                    </div>

                    <!-- Risk Flags -->
                    <div style="margin-top: 0.75rem;">
                        <div class="result-section-title" style="margin-bottom: 0.5rem; border-bottom: none; padding-bottom: 0;">Risk Flags Submitted</div>
                        <div class="risk-flags">
                            <span class="risk-flag ${requestBody.risk_of_escape ? 'active' : 'inactive'}">
                                ${requestBody.risk_of_escape ? '⚠ Flight Risk' : '✓ No Flight Risk'}
                            </span>
                            <span class="risk-flag ${requestBody.risk_of_influence ? 'active' : 'inactive'}">
                                ${requestBody.risk_of_influence ? '⚠ Witness Influence' : '✓ No Witness Risk'}
                            </span>
                            <span class="risk-flag ${requestBody.served_half_term ? 'inactive' : 'active'}">
                                ${requestBody.served_half_term ? '✓ §436A Half-Term Served' : '✗ Half-Term Not Served'}
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Section III: Likely Conditions -->
                <div class="result-section">
                    <div class="result-section-title">III. Likely Bail Conditions</div>
                    <div class="bond-badges">
                        <span class="bond-badge ${suretyLikely ? 'likely' : 'unlikely'}">
                            ${suretyLikely ? '✓' : '✗'} Surety Bond ${suretyLikely ? 'Likely Required' : 'Unlikely'}
                        </span>
                        <span class="bond-badge ${personalLikely ? 'likely' : 'unlikely'}">
                            ${personalLikely ? '✓' : '✗'} Personal Bond ${personalLikely ? 'Likely Required' : 'Unlikely'}
                        </span>
                    </div>
                </div>

                <!-- Strategy Note -->
                ${strategyNote ? `
                <div class="strategy-note ${isStatutoryWarning ? 'statutory-warning' : ''}">
                    <div class="strategy-note-title">
                        <i class="fas ${isStatutoryWarning ? 'fa-shield-alt' : 'fa-lightbulb'}"></i>
                        ${isStatutoryWarning ? 'Statutory Ground Detected' : 'Legal Strategy Note'}
                    </div>
                    <div class="strategy-note-text">${esc(strategyNote)}</div>
                </div>` : ''}

                <div class="result-timestamp">
                    Report generated: ${now} IST • Based on ${similarCases} similar case(s) in database
                </div>
            `;
        }

        // ============================================================
        // RENDER ERROR — Network failures, 500s, etc.
        // ============================================================
        function renderError(message) {
            const el = document.getElementById('result-card');
            el.innerHTML = `
                <div class="result-verdict error">
                    <div class="result-verdict-icon">❌</div>
                    <div class="result-verdict-text">ASSESSMENT FAILED</div>
                </div>
                <div class="error-detail">
                    <div class="error-detail-text">${esc(message)}</div>
                </div>
                <div class="strategy-note">
                    <div class="strategy-note-title"><i class="fas fa-redo"></i> Troubleshooting</div>
                    <div class="strategy-note-text">
                        Verify the server is running and the bail database (a.csv) is loaded correctly.
                        Check that the CSV contains columns: statute, offense_category, bail_eligibility, risk_score, surety_bond_required, personal_bond_required.
                        Review the server console for detailed error logs, then retry.
                    </div>
                </div>
            `;
        }

        // ============================================================
        // INLINE ERROR — Quick validation feedback
        // ============================================================
        function showInlineError(msg) {
            const el = document.getElementById('result-card');
            const idleEl = document.getElementById('results-idle');
            idleEl.classList.add('hidden');
            el.classList.remove('hidden');
            el.innerHTML = `
                <div class="result-verdict no-data">
                    <div class="result-verdict-icon">📝</div>
                    <div class="result-verdict-text">${esc(msg)}</div>
                </div>
            `;
        }

        // ============================================================
        // ESCAPE — XSS-safe text insertion
        // ============================================================
        function esc(s) {
            if (s === null || s === undefined) return '';
            const div = document.createElement('div');
            div.textContent = String(s);
            return div.innerHTML;
        }
    </script>
</body>
</html>
```

---

### File: `frontend/templates/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚖️ AI Legal Intelligence Platform — Supreme Court of India</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,500&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500&family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    <style>
        /* ============================================
           CSS CUSTOM PROPERTIES
           ============================================ */
        :root {
            --bg-deepest: #0a0a0f;
            --bg-dark: #0d0d14;
            --bg-panel: #111118;
            --bg-card: #16161f;
            --bg-elevated: #1a1a25;
            --bg-hover: #1e1e2a;
            --gold: #c9a84c;
            --gold-light: #e8d48b;
            --gold-dark: #8b6914;
            --gold-glow: rgba(201, 168, 76, 0.15);
            --gold-glow-strong: rgba(201, 168, 76, 0.35);
            --crimson: #8b2035;
            --crimson-light: #c4445a;
            --crimson-glow: rgba(139, 32, 53, 0.2);
            --emerald: #1a7a4c;
            --emerald-light: #4caf8a;
            --emerald-glow: rgba(26, 122, 76, 0.2);
            --amber: #b8860b;
            --amber-light: #e8b877;
            --text-primary: #e8e4dc;
            --text-secondary: #9a9488;
            --text-muted: #5a5650;
            --border-subtle: rgba(201, 168, 76, 0.08);
            --border-gold: rgba(201, 168, 76, 0.2);
            --border-strong: rgba(201, 168, 76, 0.35);
            --font-display: 'Playfair Display', serif;
            --font-body: 'Cormorant Garamond', serif;
            --font-ui: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        /* ============================================
           RESET & BASE
           ============================================ */
        *, *::before, *::after {
            margin: 0; padding: 0; box-sizing: border-box;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            background: var(--bg-deepest);
            color: var(--text-primary);
            font-family: var(--font-body);
            min-height: 100vh;
            overflow-x: hidden;
            cursor: default;
        }

        /* ============================================
           BOOT SCREEN
           ============================================ */
        .boot-screen {
            position: fixed;
            inset: 0;
            z-index: 10000;
            background: var(--bg-deepest);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            transition: opacity 0.8s ease, transform 0.8s ease;
        }

        .boot-screen.dismissed {
            opacity: 0;
            transform: scale(1.05);
            pointer-events: none;
        }

        .boot-emblem {
            position: relative;
            width: 120px;
            height: 120px;
            margin-bottom: 2rem;
        }

        .boot-emblem-ring {
            position: absolute;
            inset: 0;
            border: 2px solid var(--gold);
            border-radius: 50%;
            animation: bootRingSpin 3s linear infinite;
        }

        .boot-emblem-ring:nth-child(2) {
            inset: 10px;
            border-color: rgba(201, 168, 76, 0.4);
            animation-direction: reverse;
            animation-duration: 4s;
        }

        .boot-emblem-ring:nth-child(3) {
            inset: 20px;
            border-color: rgba(201, 168, 76, 0.2);
            animation-duration: 5s;
        }

        .boot-emblem-icon {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            animation: bootPulse 2s ease-in-out infinite;
        }

        @keyframes bootRingSpin {
            to { transform: rotate(360deg); }
        }

        @keyframes bootPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        .boot-title {
            font-family: var(--font-display);
            font-size: 1.4rem;
            color: var(--gold);
            letter-spacing: 6px;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            text-align: center;
        }

        .boot-subtitle {
            font-family: var(--font-ui);
            font-size: 0.75rem;
            color: var(--text-secondary);
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 2rem;
        }

        .boot-progress {
            width: 280px;
            height: 3px;
            background: var(--bg-card);
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }

        .boot-progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--gold-dark), var(--gold), var(--gold-light));
            border-radius: 3px;
            animation: bootLoad 3s ease-in-out forwards;
        }

        @keyframes bootLoad {
            0% { width: 0%; }
            30% { width: 40%; }
            60% { width: 70%; }
            100% { width: 100%; }
        }

        .boot-steps {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            align-items: center;
        }

        .boot-step {
            font-family: var(--font-mono);
            font-size: 0.65rem;
            color: var(--text-muted);
            letter-spacing: 1px;
            opacity: 0;
            transform: translateY(10px);
            animation: bootStepIn 0.4s ease forwards;
        }

        .boot-step .check {
            color: var(--emerald-light);
            margin-left: 0.5rem;
        }

        @keyframes bootStepIn {
            to { opacity: 1; transform: translateY(0); }
        }

        .boot-motto {
            margin-top: 2rem;
            font-family: var(--font-body);
            font-size: 0.85rem;
            color: var(--text-muted);
            font-style: italic;
        }

        /* ============================================
           ANIMATED BACKGROUND
           ============================================ */
        .bg-canvas {
            position: fixed;
            inset: 0;
            z-index: 0;
            overflow: hidden;
            pointer-events: none;
        }

        .bg-gradient-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.04;
        }

        .bg-gradient-orb.orb-1 {
            width: 800px; height: 800px;
            background: radial-gradient(circle, var(--gold) 0%, transparent 70%);
            top: -200px; left: -200px;
            animation: orbFloat1 20s ease-in-out infinite;
        }

        .bg-gradient-orb.orb-2 {
            width: 600px; height: 600px;
            background: radial-gradient(circle, var(--crimson) 0%, transparent 70%);
            bottom: -100px; right: -150px;
            animation: orbFloat2 25s ease-in-out infinite;
        }

        .bg-gradient-orb.orb-3 {
            width: 500px; height: 500px;
            background: radial-gradient(circle, var(--emerald) 0%, transparent 70%);
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            animation: orbFloat3 18s ease-in-out infinite;
        }

        @keyframes orbFloat1 {
            0%, 100% { transform: translate(0, 0); }
            33% { transform: translate(100px, 80px); }
            66% { transform: translate(-50px, 120px); }
        }

        @keyframes orbFloat2 {
            0%, 100% { transform: translate(0, 0); }
            33% { transform: translate(-80px, -60px); }
            66% { transform: translate(60px, -100px); }
        }

        @keyframes orbFloat3 {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.3); }
        }

        /* Grid lines */
        .bg-grid {
            position: fixed;
            inset: 0;
            z-index: 0;
            background-image:
                linear-gradient(rgba(201, 168, 76, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(201, 168, 76, 0.02) 1px, transparent 1px);
            background-size: 80px 80px;
            pointer-events: none;
        }

        /* Floating particles */
        .particle-field {
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            overflow: hidden;
        }

        .particle {
            position: absolute;
            width: 2px;
            height: 2px;
            background: var(--gold);
            border-radius: 50%;
            opacity: 0;
            animation: particleDrift linear infinite;
        }

        @keyframes particleDrift {
            0% { opacity: 0; transform: translateY(100vh) scale(0); }
            10% { opacity: 0.6; }
            90% { opacity: 0.6; }
            100% { opacity: 0; transform: translateY(-10vh) scale(1); }
        }

        /* ============================================
           MAIN LAYOUT
           ============================================ */
        .platform {
            position: relative;
            z-index: 1;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            opacity: 0;
            transition: opacity 0.6s ease;
        }

        .platform.visible {
            opacity: 1;
        }

        /* ============================================
           HEADER
           ============================================ */
        .header {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(10, 10, 15, 0.85);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border-bottom: 1px solid var(--border-subtle);
            padding: 0 2rem;
        }

        .header-inner {
            max-width: 1400px;
            margin: 0 auto;
            height: 70px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .brand-emblem {
            width: 42px;
            height: 42px;
            border: 1.5px solid var(--gold);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            position: relative;
        }

        .brand-emblem::before {
            content: '';
            position: absolute;
            inset: -4px;
            border: 1px solid rgba(201, 168, 76, 0.2);
            border-radius: 50%;
        }

        .brand-text h1 {
            font-family: var(--font-display);
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--gold);
            letter-spacing: 3px;
            text-transform: uppercase;
            line-height: 1.2;
        }

        .brand-text p {
            font-family: var(--font-ui);
            font-size: 0.6rem;
            color: var(--text-muted);
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .header-nav {
            display: flex;
            align-items: center;
            gap: 2rem;
        }

        .nav-indicators {
            display: flex;
            gap: 1rem;
        }

        .nav-indicator {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-family: var(--font-mono);
            font-size: 0.6rem;
            color: var(--text-muted);
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .indicator-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--emerald-light);
            box-shadow: 0 0 8px var(--emerald-light);
            animation: indicatorPulse 2s ease-in-out infinite;
        }

        @keyframes indicatorPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .header-clock {
            font-family: var(--font-mono);
            font-size: 0.7rem;
            color: var(--text-secondary);
            letter-spacing: 1px;
            padding: 0.35rem 0.8rem;
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
        }

        /* ============================================
           HERO SECTION
           ============================================ */
        .hero {
            position: relative;
            padding: 6rem 2rem 4rem;
            text-align: center;
            overflow: hidden;
        }

        .hero-ornament-top {
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--gold), transparent);
        }

        .hero-scales {
            position: relative;
            width: 200px;
            height: 160px;
            margin: 0 auto 3rem;
        }

        .scales-pillar {
            position: absolute;
            left: 50%;
            bottom: 0;
            width: 4px;
            height: 80px;
            background: linear-gradient(to top, var(--gold-dark), var(--gold));
            transform: translateX(-50%);
            border-radius: 2px;
        }

        .scales-pillar::before {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 40px;
            height: 8px;
            background: var(--gold-dark);
            border-radius: 2px;
        }

        .scales-beam-container {
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 80px;
            animation: scalesBalance 6s ease-in-out infinite;
            transform-origin: top center;
        }

        .scales-h-beam {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--gold-dark), var(--gold), var(--gold-dark));
            border-radius: 2px;
        }

        .scales-pivot-gem {
            position: absolute;
            top: -6px;
            left: 50%;
            transform: translateX(-50%);
            width: 14px;
            height: 14px;
            background: var(--gold);
            border-radius: 50%;
            box-shadow: 0 0 20px var(--gold-glow-strong);
        }

        .scales-chain {
            position: absolute;
            top: 3px;
            width: 1px;
            height: 35px;
            background: linear-gradient(to bottom, var(--gold), var(--gold-dark));
        }

        .scales-chain.left { left: 15px; }
        .scales-chain.right { right: 15px; }

        .scales-pan-group {
            position: absolute;
            top: 38px;
            width: 50px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .scales-pan-group.left { left: -10px; }
        .scales-pan-group.right { right: -10px; }

        .scales-pan-dish {
            width: 50px;
            height: 8px;
            border-radius: 0 0 25px 25px;
            background: linear-gradient(to bottom, var(--gold), var(--gold-dark));
            box-shadow: 0 3px 12px rgba(201, 168, 76, 0.2);
        }

        .scales-pan-label {
            font-family: var(--font-mono);
            font-size: 0.45rem;
            letter-spacing: 2px;
            color: var(--text-muted);
            margin-top: 6px;
            text-transform: uppercase;
        }

        @keyframes scalesBalance {
            0%, 100% { transform: translateX(-50%) rotate(0deg); }
            25% { transform: translateX(-50%) rotate(2deg); }
            75% { transform: translateX(-50%) rotate(-2deg); }
        }

        .hero-title {
            font-family: var(--font-display);
            font-size: 3.2rem;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1.15;
            margin-bottom: 1rem;
            letter-spacing: 1px;
        }

        .hero-title .gold {
            color: var(--gold);
            position: relative;
        }

        .hero-title .gold::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--gold), transparent);
        }

        .hero-subtitle {
            font-family: var(--font-body);
            font-size: 1.3rem;
            color: var(--text-secondary);
            max-width: 700px;
            margin: 0 auto 1.5rem;
            line-height: 1.6;
            font-weight: 400;
        }

        .hero-divider {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin: 2rem auto;
        }

        .hero-divider-line {
            width: 80px;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-gold));
        }

        .hero-divider-line.right {
            background: linear-gradient(90deg, var(--border-gold), transparent);
        }

        .hero-divider-diamond {
            color: var(--gold);
            font-size: 0.7rem;
        }

        .hero-tech-badges {
            display: flex;
            justify-content: center;
            gap: 0.8rem;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }

        .tech-badge {
            font-family: var(--font-mono);
            font-size: 0.6rem;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            padding: 0.35rem 0.8rem;
            border: 1px solid var(--border-subtle);
            border-radius: 3px;
            color: var(--text-muted);
            background: rgba(201, 168, 76, 0.03);
        }

        .hero-motto {
            font-family: var(--font-body);
            font-size: 1rem;
            color: var(--text-muted);
            font-style: italic;
            margin-top: 1rem;
        }

        /* ============================================
           CARDS SECTION
           ============================================ */
        .cards-section {
            padding: 2rem 2rem 6rem;
            max-width: 1200px;
            margin: 0 auto;
        }

        .section-label {
            text-align: center;
            margin-bottom: 3rem;
        }

        .section-label-text {
            font-family: var(--font-mono);
            font-size: 0.65rem;
            letter-spacing: 4px;
            text-transform: uppercase;
            color: var(--gold);
            padding: 0.4rem 1.2rem;
            border: 1px solid var(--border-gold);
            border-radius: 2px;
            display: inline-block;
        }

        .cards-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2.5rem;
        }

        /* ============================================
           TOOL CARD
           ============================================ */
        .tool-card {
            position: relative;
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
            text-decoration: none;
            color: inherit;
            display: block;
        }

        .tool-card:hover {
            border-color: var(--border-gold);
            transform: translateY(-8px);
            box-shadow:
                0 20px 60px rgba(0, 0, 0, 0.4),
                0 0 80px var(--gold-glow),
                inset 0 1px 0 rgba(201, 168, 76, 0.1);
        }

        .tool-card::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, var(--gold-glow) 0%, transparent 50%);
            opacity: 0;
            transition: opacity 0.5s ease;
            z-index: 0;
        }

        .tool-card:hover::before {
            opacity: 1;
        }

        /* Shimmer effect on hover */
        .tool-card::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(201, 168, 76, 0.05),
                transparent
            );
            transition: none;
            z-index: 0;
        }

        .tool-card:hover::after {
            animation: cardShimmer 1.5s ease forwards;
        }

        @keyframes cardShimmer {
            to { left: 100%; }
        }

        .card-glow-border {
            position: absolute;
            inset: -1px;
            border-radius: 17px;
            padding: 1px;
            background: linear-gradient(135deg, var(--gold), transparent 40%, transparent 60%, var(--gold-dark));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            opacity: 0;
            transition: opacity 0.5s ease;
            z-index: 0;
        }

        .tool-card:hover .card-glow-border {
            opacity: 1;
        }

        .card-header {
            position: relative;
            z-index: 1;
            padding: 2.5rem 2.5rem 0;
        }

        .card-icon-area {
            position: relative;
            width: 80px;
            height: 80px;
            margin-bottom: 1.5rem;
        }

        .card-icon-ring {
            position: absolute;
            inset: 0;
            border: 1.5px solid var(--border-gold);
            border-radius: 50%;
            transition: all 0.5s ease;
        }

        .tool-card:hover .card-icon-ring {
            border-color: var(--gold);
            box-shadow: 0 0 30px var(--gold-glow);
            transform: scale(1.05);
        }

        .card-icon-ring::before {
            content: '';
            position: absolute;
            inset: 6px;
            border: 1px solid rgba(201, 168, 76, 0.15);
            border-radius: 50%;
        }

        .card-icon-symbol {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            transition: transform 0.5s ease;
        }

        .tool-card:hover .card-icon-symbol {
            transform: scale(1.15) rotate(5deg);
        }

        .card-number {
            font-family: var(--font-mono);
            font-size: 0.55rem;
            letter-spacing: 3px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .card-title {
            font-family: var(--font-display);
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.75rem;
            line-height: 1.2;
            transition: color 0.3s ease;
        }

        .tool-card:hover .card-title {
            color: var(--gold);
        }

        .card-subtitle {
            font-family: var(--font-ui);
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.7;
            margin-bottom: 1.5rem;
        }

        .card-body {
            position: relative;
            z-index: 1;
            padding: 0 2.5rem;
        }

        .card-features {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            margin-bottom: 1.5rem;
        }

        .card-feature {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            font-family: var(--font-ui);
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .card-feature i {
            color: var(--gold);
            font-size: 0.6rem;
            width: 16px;
            text-align: center;
        }

        .card-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            padding: 1.2rem 0;
            border-top: 1px solid var(--border-subtle);
            margin-bottom: 1.5rem;
        }

        .card-stat {
            text-align: center;
        }

        .card-stat-number {
            font-family: var(--font-display);
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--gold);
            line-height: 1.2;
        }

        .card-stat-label {
            font-family: var(--font-mono);
            font-size: 0.5rem;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 0.2rem;
        }

        .card-footer {
            position: relative;
            z-index: 1;
            padding: 1.5rem 2.5rem 2.5rem;
        }

        .card-cta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.5rem;
            background: rgba(201, 168, 76, 0.05);
            border: 1px solid var(--border-gold);
            border-radius: 8px;
            transition: all 0.4s ease;
        }

        .tool-card:hover .card-cta {
            background: rgba(201, 168, 76, 0.1);
            border-color: var(--gold);
            box-shadow: 0 0 30px var(--gold-glow);
        }

        .cta-text {
            font-family: var(--font-display);
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--gold);
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .cta-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: var(--gold);
            color: var(--bg-deepest);
            font-size: 0.85rem;
            transition: all 0.4s ease;
        }

        .tool-card:hover .cta-arrow {
            transform: translateX(4px);
            box-shadow: 0 0 20px var(--gold-glow-strong);
        }

        /* Card-specific accent colors */
        .tool-card.citation-card:hover {
            border-color: rgba(201, 168, 76, 0.4);
        }

        .tool-card.bail-card:hover {
            border-color: rgba(76, 175, 138, 0.4);
        }

        .tool-card.bail-card .card-icon-ring {
            border-color: rgba(76, 175, 138, 0.2);
        }

        .tool-card.bail-card:hover .card-icon-ring {
            border-color: var(--emerald-light);
            box-shadow: 0 0 30px var(--emerald-glow);
        }

        .tool-card.bail-card:hover .card-title {
            color: var(--emerald-light);
        }

        .tool-card.bail-card .card-stat-number {
            color: var(--emerald-light);
        }

        .tool-card.bail-card .card-feature i {
            color: var(--emerald-light);
        }

        .tool-card.bail-card .cta-text {
            color: var(--emerald-light);
        }

        .tool-card.bail-card .cta-arrow {
            background: var(--emerald-light);
        }

        .tool-card.bail-card:hover .card-cta {
            background: rgba(76, 175, 138, 0.08);
            border-color: var(--emerald-light);
            box-shadow: 0 0 30px var(--emerald-glow);
        }

        .tool-card.bail-card::before {
            background: linear-gradient(135deg, var(--emerald-glow) 0%, transparent 50%);
        }

        .tool-card.bail-card .card-glow-border {
            background: linear-gradient(135deg, var(--emerald-light), transparent 40%, transparent 60%, var(--emerald));
        }

        /* ============================================
           CAPABILITIES RIBBON
           ============================================ */
        .capabilities-ribbon {
            padding: 3rem 2rem;
            border-top: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            background: rgba(201, 168, 76, 0.015);
        }

        .ribbon-inner {
            max-width: 1200px;
            margin: 0 auto;
        }

        .ribbon-title {
            text-align: center;
            font-family: var(--font-mono);
            font-size: 0.6rem;
            letter-spacing: 4px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 2rem;
        }

        .ribbon-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
        }

        .ribbon-item {
            text-align: center;
            padding: 1.5rem 1rem;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            background: var(--bg-panel);
            transition: all 0.3s ease;
        }

        .ribbon-item:hover {
            border-color: var(--border-gold);
            background: var(--bg-card);
            transform: translateY(-2px);
        }

        .ribbon-icon {
            font-size: 1.5rem;
            margin-bottom: 0.75rem;
        }

        .ribbon-item-title {
            font-family: var(--font-ui);
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.4rem;
        }

        .ribbon-item-desc {
            font-family: var(--font-ui);
            font-size: 0.65rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        /* ============================================
           FOOTER
           ============================================ */
        .footer {
            padding: 2rem;
            text-align: center;
            border-top: 1px solid var(--border-subtle);
        }

        .footer-ornament {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .footer-ornament-line {
            width: 60px;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-gold));
        }

        .footer-ornament-line.right {
            background: linear-gradient(90deg, var(--border-gold), transparent);
        }

        .footer-motto {
            font-family: var(--font-body);
            font-size: 1rem;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
        }

        .footer-motto-en {
            font-family: var(--font-ui);
            font-size: 0.65rem;
            color: var(--text-muted);
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .footer-powered {
            margin-top: 1.5rem;
            font-family: var(--font-mono);
            font-size: 0.55rem;
            color: var(--text-muted);
            letter-spacing: 1.5px;
        }

        /* ============================================
           MOUSE GLOW FOLLOWER
           ============================================ */
        .mouse-glow {
            position: fixed;
            width: 400px;
            height: 400px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(201, 168, 76, 0.06) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
            transition: transform 0.15s ease;
            transform: translate(-50%, -50%);
        }

        /* ============================================
           RESPONSIVE
           ============================================ */
        @media (max-width: 900px) {
            .cards-grid {
                grid-template-columns: 1fr;
                gap: 2rem;
            }

            .hero-title {
                font-size: 2.2rem;
            }

            .hero-subtitle {
                font-size: 1.1rem;
            }

            .ribbon-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .header-nav {
                gap: 1rem;
            }

            .nav-indicators {
                display: none;
            }
        }

        @media (max-width: 600px) {
            .hero {
                padding: 4rem 1.5rem 3rem;
            }

            .hero-title {
                font-size: 1.8rem;
            }

            .card-header, .card-body, .card-footer {
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }

            .card-stats {
                grid-template-columns: repeat(2, 1fr);
            }

            .ribbon-grid {
                grid-template-columns: 1fr;
            }

            .header-inner {
                padding: 0 1rem;
            }
        }

        /* ============================================
           SCROLL REVEAL ANIMATIONS
           ============================================ */
        .reveal {
            opacity: 0;
            transform: translateY(40px);
            transition: all 0.8s cubic-bezier(0.23, 1, 0.32, 1);
        }

        .reveal.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .reveal-delay-1 { transition-delay: 0.15s; }
        .reveal-delay-2 { transition-delay: 0.3s; }
        .reveal-delay-3 { transition-delay: 0.45s; }
        .reveal-delay-4 { transition-delay: 0.6s; }

        /* ============================================
           TYPING ANIMATION FOR HERO
           ============================================ */
        .typing-cursor {
            display: inline-block;
            width: 2px;
            height: 1em;
            background: var(--gold);
            margin-left: 4px;
            animation: cursorBlink 1s step-end infinite;
            vertical-align: text-bottom;
        }

        @keyframes cursorBlink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }

        /* Spinning courthouse for card decoration */
        .card-bg-pattern {
            position: absolute;
            top: 0;
            right: 0;
            width: 200px;
            height: 200px;
            opacity: 0.03;
            z-index: 0;
            transition: opacity 0.5s ease;
        }

        .tool-card:hover .card-bg-pattern {
            opacity: 0.06;
        }

        .card-bg-pattern svg {
            width: 100%;
            height: 100%;
        }

        /* Status bar animation */
        .live-counter {
            display: inline-block;
            font-family: var(--font-mono);
            color: var(--gold);
            font-variant-numeric: tabular-nums;
        }
    </style>
</head>
<body>

    <!-- ========== BOOT SCREEN ========== -->
    <div class="boot-screen" id="boot-screen">
        <div class="boot-emblem">
            <div class="boot-emblem-ring"></div>
            <div class="boot-emblem-ring"></div>
            <div class="boot-emblem-ring"></div>
            <div class="boot-emblem-icon">⚖️</div>
        </div>
        <div class="boot-title">Supreme Court of India</div>
        <div class="boot-subtitle">AI Legal Intelligence Platform</div>
        <div class="boot-progress">
            <div class="boot-progress-fill"></div>
        </div>
        <div class="boot-steps">
            <div class="boot-step" style="animation-delay: 0.3s">§ Initializing Neural Legal Engine... <span class="check">✓</span></div>
            <div class="boot-step" style="animation-delay: 0.8s">§ Loading Case Archive & Bail Reckoner... <span class="check">✓</span></div>
            <div class="boot-step" style="animation-delay: 1.3s">§ Connecting to Groq LLM × MiniLM-L6 RAG... <span class="check">✓</span></div>
            <div class="boot-step" style="animation-delay: 1.8s">§ Engaging Hallucination Detection Protocol... <span class="check">✓</span></div>
            <div class="boot-step" style="animation-delay: 2.3s">§ Upholding the Rule of Law... <span class="check">✓</span></div>
        </div>
        <div class="boot-motto">"Yato Dharmastato Jayah"</div>
    </div>

    <!-- ========== ANIMATED BACKGROUND ========== -->
    <div class="bg-canvas">
        <div class="bg-gradient-orb orb-1"></div>
        <div class="bg-gradient-orb orb-2"></div>
        <div class="bg-gradient-orb orb-3"></div>
    </div>
    <div class="bg-grid"></div>
    <div class="particle-field" id="particle-field"></div>
    <div class="mouse-glow" id="mouse-glow"></div>

    <!-- ========== MAIN PLATFORM ========== -->
    <div class="platform" id="platform">

        <!-- HEADER -->
        <header class="header">
            <div class="header-inner">
                <div class="header-brand">
                    <div class="brand-emblem">⚖️</div>
                    <div class="brand-text">
                        <h1>Legal AI Platform</h1>
                        <p>Supreme Court of India</p>
                    </div>
                </div>
                <div class="header-nav">
                    <div class="nav-indicators">
                        <div class="nav-indicator">
                            <div class="indicator-dot"></div>
                            <span>Groq API</span>
                        </div>
                        <div class="nav-indicator">
                            <div class="indicator-dot"></div>
                            <span>LLM Active</span>
                        </div>
                        <div class="nav-indicator">
                            <div class="indicator-dot"></div>
                            <span>RAG Engine</span>
                        </div>
                    </div>
                    <div class="header-clock" id="header-clock">—</div>
                </div>
            </div>
        </header>

        <!-- HERO -->
        <section class="hero">
            <div class="hero-ornament-top"></div>

            <div class="hero-scales">
                <div class="scales-pillar"></div>
                <div class="scales-beam-container">
                    <div class="scales-h-beam"></div>
                    <div class="scales-pivot-gem"></div>
                    <div class="scales-chain left"></div>
                    <div class="scales-chain right"></div>
                    <div class="scales-pan-group left">
                        <div class="scales-pan-dish"></div>
                        <div class="scales-pan-label">Truth</div>
                    </div>
                    <div class="scales-pan-group right">
                        <div class="scales-pan-dish"></div>
                        <div class="scales-pan-label">Justice</div>
                    </div>
                </div>
            </div>

            <h2 class="hero-title reveal">
                Artificial Intelligence for<br>
                <span class="gold">Indian Legal Practice</span>
            </h2>

            <p class="hero-subtitle reveal reveal-delay-1">
                Harness the power of AI to verify citations, detect fabricated precedents, 
                verify judicial quotes, and calculate bail eligibility — all powered by 
                cutting-edge language models and retrieval-augmented generation.
            </p>

            <div class="hero-divider reveal reveal-delay-2">
                <div class="hero-divider-line"></div>
                <div class="hero-divider-diamond">◆</div>
                <div class="hero-divider-line right"></div>
            </div>

            <div class="hero-tech-badges reveal reveal-delay-2">
                <span class="tech-badge">Groq Inference</span>
                <span class="tech-badge">Llama 3.3 70B</span>
                <span class="tech-badge">MiniLM-L6 RAG</span>
                <span class="tech-badge">Hallucination Detection</span>
                <span class="tech-badge">Quote Verification</span>
            </div>

            <div class="hero-motto reveal reveal-delay-3">
                "यतो धर्मस्ततो जयः" — Where there is Righteousness, there is Victory
            </div>
        </section>

        <!-- TOOL CARDS -->
        <section class="cards-section">
            <div class="section-label reveal">
                <span class="section-label-text">§ Select Your Chamber §</span>
            </div>

            <div class="cards-grid">
                <!-- CITATION AUDITOR CARD -->
                <a href="/auditor" class="tool-card citation-card reveal reveal-delay-1" id="card-citation">
                    <div class="card-glow-border"></div>
                    <div class="card-bg-pattern">
                        <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="100" cy="100" r="80" stroke="currentColor" stroke-width="0.5"/>
                            <circle cx="100" cy="100" r="60" stroke="currentColor" stroke-width="0.5"/>
                            <circle cx="100" cy="100" r="40" stroke="currentColor" stroke-width="0.5"/>
                            <line x1="100" y1="20" x2="100" y2="180" stroke="currentColor" stroke-width="0.3"/>
                            <line x1="20" y1="100" x2="180" y2="100" stroke="currentColor" stroke-width="0.3"/>
                        </svg>
                    </div>
                    <div class="card-header">
                        <div class="card-icon-area">
                            <div class="card-icon-ring"></div>
                            <div class="card-icon-symbol">🔍</div>
                        </div>
                        <div class="card-number">Module I — Verification Engine</div>
                        <h3 class="card-title">Citation Auditor</h3>
                        <p class="card-subtitle">
                            Upload any legal document and our AI will extract every cited case, 
                            verify it against the Supreme Court archive, and use RAG to check 
                            if attributed quotes are accurate — catching hallucinated citations 
                            before they reach the bench.
                        </p>
                    </div>
                    <div class="card-body">
                        <div class="card-features">
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>AI-powered citation extraction from PDFs</span>
                            </div>
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>Cross-reference against SC case archive</span>
                            </div>
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>RAG quote verification with source judgments</span>
                            </div>
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>Hallucination detection & confidence scoring</span>
                            </div>
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>Bulk document audit & export reports</span>
                            </div>
                        </div>
                        <div class="card-stats">
                            <div class="card-stat">
                                <div class="card-stat-number" id="stat-cases">—</div>
                                <div class="card-stat-label">Cases in Registry</div>
                            </div>
                            <div class="card-stat">
                                <div class="card-stat-number">70B</div>
                                <div class="card-stat-label">LLM Parameters</div>
                            </div>
                            <div class="card-stat">
                                <div class="card-stat-number">RAG</div>
                                <div class="card-stat-label">Quote Engine</div>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer">
                        <div class="card-cta">
                            <span class="cta-text">Enter Audit Chamber</span>
                            <div class="cta-arrow">
                                <i class="fas fa-arrow-right"></i>
                            </div>
                        </div>
                    </div>
                </a>

                <!-- BAIL RECKONER CARD -->
                <a href="/bail-reckoner" class="tool-card bail-card reveal reveal-delay-2" id="card-bail">
                    <div class="card-glow-border"></div>
                    <div class="card-bg-pattern">
                        <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <rect x="40" y="40" width="120" height="120" rx="10" stroke="currentColor" stroke-width="0.5"/>
                            <rect x="60" y="60" width="80" height="80" rx="6" stroke="currentColor" stroke-width="0.5"/>
                            <rect x="80" y="80" width="40" height="40" rx="4" stroke="currentColor" stroke-width="0.5"/>
                            <line x1="40" y1="100" x2="160" y2="100" stroke="currentColor" stroke-width="0.3"/>
                            <line x1="100" y1="40" x2="100" y2="160" stroke="currentColor" stroke-width="0.3"/>
                        </svg>
                    </div>
                    <div class="card-header">
                        <div class="card-icon-area">
                            <div class="card-icon-ring"></div>
                            <div class="card-icon-symbol">⚖️</div>
                        </div>
                        <div class="card-number">Module II — Risk Assessment Engine</div>
                        <h3 class="card-title">Bail Reckoner</h3>
                        <p class="card-subtitle">
                            Calculate bail eligibility using AI-driven risk assessment. 
                            Our engine analyzes historical case data, statutory provisions 
                            under CrPC, PMLA, and NDPS to provide probability scores, 
                            bond recommendations, and legal strategy insights.
                        </p>
                    </div>
                    <div class="card-body">
                        <div class="card-features">
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>Historical case-based probability analysis</span>
                            </div>
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>Risk scoring (flight risk, influence, recidivism)</span>
                            </div>
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>CrPC Section 436A default bail detection</span>
                            </div>
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>Surety & personal bond requirement prediction</span>
                            </div>
                            <div class="card-feature">
                                <i class="fas fa-check"></i>
                                <span>Strategic legal recommendations</span>
                            </div>
                        </div>
                        <div class="card-stats">
                            <div class="card-stat">
                                <div class="card-stat-number" id="stat-bail-cases">—</div>
                                <div class="card-stat-label">Bail Records</div>
                            </div>
                            <div class="card-stat">
                                <div class="card-stat-number">AI</div>
                                <div class="card-stat-label">Risk Engine</div>
                            </div>
                            <div class="card-stat">
                                <div class="card-stat-number">§436A</div>
                                <div class="card-stat-label">CrPC Aware</div>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer">
                        <div class="card-cta">
                            <span class="cta-text">Enter Bail Chamber</span>
                            <div class="cta-arrow">
                                <i class="fas fa-arrow-right"></i>
                            </div>
                        </div>
                    </div>
                </a>
            </div>
        </section>

        <!-- CAPABILITIES RIBBON -->
        <section class="capabilities-ribbon">
            <div class="ribbon-inner">
                <div class="ribbon-title">Platform Capabilities</div>
                <div class="ribbon-grid">
                    <div class="ribbon-item reveal">
                        <div class="ribbon-icon">🧠</div>
                        <div class="ribbon-item-title">LLM Intelligence</div>
                        <div class="ribbon-item-desc">Powered by Llama 3.3 70B via Groq for sub-second legal reasoning</div>
                    </div>
                    <div class="ribbon-item reveal reveal-delay-1">
                        <div class="ribbon-icon">📚</div>
                        <div class="ribbon-item-title">RAG Pipeline</div>
                        <div class="ribbon-item-desc">MiniLM-L6 embeddings search source judgments to verify quoted text</div>
                    </div>
                    <div class="ribbon-item reveal reveal-delay-2">
                        <div class="ribbon-icon">🔬</div>
                        <div class="ribbon-item-title">Hallucination Guard</div>
                        <div class="ribbon-item-desc">Multi-layer verification: fuzzy search → LLM match → RAG quote check</div>
                    </div>
                    <div class="ribbon-item reveal reveal-delay-3">
                        <div class="ribbon-icon">📊</div>
                        <div class="ribbon-item-title">Data-Driven Bail</div>
                        <div class="ribbon-item-desc">Historical bail outcomes analyzed for probability scoring & risk assessment</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- FOOTER -->
        <footer class="footer">
            <div class="footer-ornament">
                <div class="footer-ornament-line"></div>
                <span style="color: var(--gold); font-size: 0.7rem;">◆</span>
                <div class="footer-ornament-line right"></div>
            </div>
            <div class="footer-motto">यतो धर्मस्ततो जयः</div>
            <div class="footer-motto-en">Where there is Righteousness, there is Victory</div>
            <div class="footer-powered">
                POWERED BY GROQ × LLAMA 3.3 70B × MINILM-L6 RAG × SUPREME COURT ARCHIVE
            </div>
        </footer>
    </div>

    <!-- ========== JAVASCRIPT ========== -->
    <script>
        // ============================
        // CONFIG
        // ============================
        const API_BASE = window.location.hostname === 'localhost' 
            ? 'http://localhost:8000' 
            : '';

        // ============================
        // BOOT SEQUENCE
        // ============================
        const bootScreen = document.getElementById('boot-screen');
        const platform = document.getElementById('platform');

        function runBoot() {
            // Fetch stats during boot
            fetchPlatformStats();

            setTimeout(() => {
                bootScreen.classList.add('dismissed');
                platform.classList.add('visible');
                setTimeout(() => {
                    bootScreen.style.display = 'none';
                    initScrollReveal();
                }, 800);
            }, 3500);
        }

        async function fetchPlatformStats() {
            try {
                const resp = await fetch(`${API_BASE}/db-stats`);
                const data = await resp.json();
                if (data.loaded && data.record_count) {
                    animateCounter('stat-cases', data.record_count);
                }
            } catch (e) {
                console.warn('Backend not reachable during boot:', e);
            }

            // Try to get bail reckoner stats
            try {
                document.getElementById('stat-bail-cases').textContent = 'LIVE';
            } catch (e) {}
        }

        function animateCounter(elementId, target) {
            const el = document.getElementById(elementId);
            if (!el) return;

            const duration = 2000;
            const startTime = performance.now();
            const startVal = 0;

            function update(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                // Ease out cubic
                const eased = 1 - Math.pow(1 - progress, 3);
                const current = Math.round(startVal + (target - startVal) * eased);

                if (target >= 1000) {
                    el.textContent = (current / 1000).toFixed(1) + 'K';
                } else {
                    el.textContent = current.toLocaleString();
                }

                if (progress < 1) {
                    requestAnimationFrame(update);
                }
            }

            requestAnimationFrame(update);
        }

        // ============================
        // CLOCK
        // ============================
        function updateClock() {
            const now = new Date();
            const dateStr = now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
            const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
            const el = document.getElementById('header-clock');
            if (el) el.textContent = `${dateStr} ${timeStr}`;
        }
        setInterval(updateClock, 1000);
        updateClock();

        // ============================
        // PARTICLES
        // ============================
        function createParticles() {
            const field = document.getElementById('particle-field');
            const count = 40;

            for (let i = 0; i < count; i++) {
                const p = document.createElement('div');
                p.className = 'particle';
                p.style.left = Math.random() * 100 + '%';
                p.style.animationDuration = (8 + Math.random() * 15) + 's';
                p.style.animationDelay = (Math.random() * 10) + 's';
                p.style.width = (1 + Math.random() * 2) + 'px';
                p.style.height = p.style.width;
                field.appendChild(p);
            }
        }

        // ============================
        // MOUSE GLOW
        // ============================
        const mouseGlow = document.getElementById('mouse-glow');
        let mouseX = -500, mouseY = -500;

        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        function updateMouseGlow() {
            mouseGlow.style.transform = `translate(${mouseX - 200}px, ${mouseY - 200}px)`;
            requestAnimationFrame(updateMouseGlow);
        }

        // ============================
        // SCROLL REVEAL
        // ============================
        function initScrollReveal() {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                    }
                });
            }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

            document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
        }

        // ============================
        // CARD TILT EFFECT
        // ============================
        function initCardTilt() {
            document.querySelectorAll('.tool-card').forEach(card => {
                card.addEventListener('mousemove', (e) => {
                    const rect = card.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    const centerX = rect.width / 2;
                    const centerY = rect.height / 2;
                    const rotateX = ((y - centerY) / centerY) * -3;
                    const rotateY = ((x - centerX) / centerX) * 3;

                    card.style.transform = `translateY(-8px) perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
                });

                card.addEventListener('mouseleave', () => {
                    card.style.transform = 'translateY(0) perspective(1000px) rotateX(0) rotateY(0)';
                });
            });
        }

        // ============================
        // KEYBOARD SHORTCUTS
        // ============================
        document.addEventListener('keydown', (e) => {
            if (e.key === '1' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                window.location.href = '/auditor';
            }
            if (e.key === '2' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                window.location.href = '/bail-reckoner';
            }
        });

        // ============================
        // INIT
        // ============================
        document.addEventListener('DOMContentLoaded', () => {
            createParticles();
            updateMouseGlow();
            runBoot();
            setTimeout(initCardTilt, 4000);
        });
    </script>
</body>
</html>
```

---

