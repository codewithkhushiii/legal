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
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
    
# ==========================================
# 6. Case Research / Similar Case Finder
# ==========================================

# --- Pydantic Models for Case Research ---
class CaseResearchRequest(BaseModel):
    case_description: str
    legal_domain: Optional[str] = None
    key_statutes: Optional[List[str]] = []
    num_results: Optional[int] = 5

class CaseDetailRequest(BaseModel):
    source_file: str

# --- Global state for case intelligence layer ---
case_cards_df = pd.DataFrame()
case_embeddings_matrix = None

def load_case_intelligence():
    """Load the case cards and embeddings built by build_case_index.py"""
    global case_cards_df, case_embeddings_matrix

    cards_path = Path("case_cards.parquet")
    embeddings_path = Path("case_embeddings.npy")

    if cards_path.exists():
        try:
            case_cards_df = pd.read_parquet(cards_path)
            logger.info(f"📚 Loaded {len(case_cards_df)} case cards from {cards_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load case cards: {e}")

    if embeddings_path.exists():
        try:
            case_embeddings_matrix = np.load(embeddings_path)
            logger.info(
                f"🧠 Loaded case embeddings: {case_embeddings_matrix.shape} "
                f"from {embeddings_path}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not load case embeddings: {e}")


# --- Call this inside your lifespan function ---
# Add this line inside the lifespan() async context manager, 
# right after loading bail_df:
#
#   load_case_intelligence()



@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, bail_df
    print("🚀 Starting Server: Gathering metadata from all folders...")
    all_dataframes = []

    for file_path in Path('.').rglob('*.parquet'):
        if 'venv' in file_path.parts:
            continue
        # Skip the case_cards.parquet — it's loaded separately
        if file_path.name == 'case_cards.parquet':
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

    # Load Case Intelligence Layer
    load_case_intelligence()

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
class VoiceAnalyzeRequest(BaseModel):
    transcript: str
    language: Optional[str] = "english"

class ChatMessage(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    audit_context: Optional[str] = None
    language: Optional[str] = "english"

class CitationRequest(BaseModel):
    citation: str
    language: Optional[str] = "english"

class SummaryRequest(BaseModel):
    results: list
    total: int
    sc_count: int
    hc_count: int
    language: Optional[str] = "english"
    
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

# Voice Assistant route
@app.get("/voice-assistant", response_class=FileResponse)
async def serve_voice_assistant():
    return FileResponse("frontend/templates/voice-assistant.html")

# API stats endpoint (keep your existing one, just rename the root)
@app.get("/api/health")
def api_health():
    return {
        "message": "Legal AI Platform API is running!",
        "endpoints": ["/audit-document", "/audit-multiple", "/verify-citation", "/chat", "/summarize", "/reckoner/bail", "/db-stats", "/voice-analyze"],
        "docs": "/docs"
    }

# ==========================================
# 4. Core Helper Functions
# ==========================================
def chunk_text(text, chunk_size=30000, overlap=3000):
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



# ==========================================
# 5. Language Support Helper
# ==========================================
def get_language_prompt_suffix(language: str = "english") -> str:
    """Returns language-specific instructions for the LLM."""
    language_instructions = {
        "english": "Respond in English. Be professional and precise.",
        "hindi": "Respond in Hindi (Devanagari script). Use formal legal terminology.",
        "hinglish": "Respond in Hinglish (Hindi words in Roman script mixed with English). Keep it professional."
    }
    return language_instructions.get(language.lower(), language_instructions["english"])

# ==========================================
# 6. Core Audit Functions with Language Support
# ==========================================

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


def verify_quotation(attributed_claim: str, source_file_path: str, language: str = "english"):
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
    lang_suffix = get_language_prompt_suffix(language)
    
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

{lang_suffix}

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


def extract_citations_with_groq(full_text, language: str = "english"):
    print("🧠 Splitting document into manageable chunks...")
    chunks = chunk_text(full_text)
    
    all_extracted_cases = {}
    lang_suffix = get_language_prompt_suffix(language)

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
        
        {lang_suffix}
        
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


def resolve_match_with_llm(target_citation, candidates_df, language: str = "english"):
    if candidates_df.empty:
        return {"status": "🔴 NO CANDIDATES", "message": "No similar cases found in database.", "confidence": 0}

    candidate_dict = {
        str(row['index']): f"{row.get('title', 'Unknown')} (NC: {row.get('nc_display', 'N/A')}, Citation: {row.get('citation', 'N/A')})"
        for _, row in candidates_df.iterrows()
    }

    lang_suffix = get_language_prompt_suffix(language)

    prompt = f"""
    You are a STRICT legal AI auditor. Verify if a Target Citation exists in the Database Candidates.
    Target Citation: "{target_citation}"
    Database Candidates: {json.dumps(candidate_dict)}
    
    RULES:
    1. Minor formatting differences ("v." vs "versus") or missing articles ("The") are ALLOWED.
    2. If the neutral citation (NC) matches (e.g. 2024 INSC 2), it is an EXACT match.
    3. If party names match on both sides (petitioner vs respondent), it is an EXACT match.
    If there is an EXACT match, return its ID. If NO EXACT MATCH, return "null".
    
    {lang_suffix}
    
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


def batch_resolve_matches_with_llm(citation_candidate_pairs: list, language: str = "english"):
    """
    Batch-resolve multiple citations in a SINGLE LLM call to reduce API overhead.
    
    Args:
        citation_candidate_pairs: list of (citation_name, candidates_df) tuples
        language: language for response
    
    Returns:
        dict mapping citation_name -> verification result
    """
    # Separate citations with candidates vs without
    results = {}
    batch_items = []
    
    for citation, candidates_df in citation_candidate_pairs:
        if candidates_df.empty:
            results[citation] = {"status": "🔴 NO CANDIDATES", "message": "No similar cases found in database.", "confidence": 0}
        else:
            candidate_dict = {
                str(row['index']): f"{row.get('title', 'Unknown')} (NC: {row.get('nc_display', 'N/A')}, Citation: {row.get('citation', 'N/A')})"
                for _, row in candidates_df.iterrows()
            }
            batch_items.append((citation, candidate_dict))
    
    if not batch_items:
        return results
    
    # Process in batches of up to 5 citations per LLM call
    BATCH_SIZE = 5
    lang_suffix = get_language_prompt_suffix(language)
    
    for batch_start in range(0, len(batch_items), BATCH_SIZE):
        batch = batch_items[batch_start:batch_start + BATCH_SIZE]
        
        # Build a combined prompt for all citations in this batch
        citations_section = ""
        for idx, (citation, candidate_dict) in enumerate(batch, 1):
            citations_section += f"""
CITATION #{idx}: "{citation}"
Candidates: {json.dumps(candidate_dict)}

"""
        
        prompt = f"""You are a STRICT legal AI auditor. Verify if EACH of the following Target Citations exists in their respective Database Candidates.

{citations_section}

RULES (apply to ALL citations):
1. Minor formatting differences ("v." vs "versus") or missing articles ("The") are ALLOWED.
2. If the neutral citation (NC) matches (e.g. 2024 INSC 2), it is an EXACT match.
3. If party names match on both sides (petitioner vs respondent), it is an EXACT match.
If there is an EXACT match, return its ID. If NO EXACT MATCH, return "null".

{lang_suffix}

Respond ONLY with JSON containing a "results" array with one entry per citation in ORDER:
{{"results": [{{"citation_number": 1, "matched_id": "id_or_null", "reason": "Short reason", "confidence": 85}}, ...]}}
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
            batch_results = result_json.get("results", [])
            
            for idx, (citation, candidate_dict) in enumerate(batch):
                if idx < len(batch_results):
                    item = batch_results[idx]
                else:
                    results[citation] = {"status": "ERROR", "message": "No result from batch LLM call", "confidence": 0}
                    continue
                
                matched_id = item.get("matched_id")
                reason = item.get("reason", "No reason provided.")
                confidence = item.get("confidence", 50)
                
                if matched_id and str(matched_id) != "null":
                    try:
                        winning_row = df[df['index'] == int(matched_id)].iloc[0]
                        file_path_val = winning_row.get('path', '')
                        results[citation] = {
                            "status": "🟢 VERIFIED BY AI",
                            "matched_name": winning_row.get('title', 'Unknown'),
                            "matched_citation": winning_row.get('nc_display', winning_row.get('citation', 'N/A')),
                            "file_to_open": str(file_path_val) if pd.notna(file_path_val) else "",
                            "reason": reason,
                            "confidence": confidence
                        }
                    except (ValueError, IndexError) as e:
                        logger.error(f"Could not find matched_id {matched_id} for '{citation}': {e}")
                        results[citation] = {
                            "status": "🔴 HALLUCINATION DETECTED",
                            "message": f"LLM returned invalid ID: {matched_id}",
                            "confidence": 0
                        }
                else:
                    results[citation] = {
                        "status": "🔴 HALLUCINATION DETECTED",
                        "message": f"Reason: {reason}",
                        "confidence": confidence
                    }
                    
        except Exception as e:
            logger.error(f"Batch LLM resolution failed: {e}")
            # Fall back to individual resolution for this batch
            for citation, candidate_dict in batch:
                if citation not in results:
                    # Re-get candidates and use single resolution
                    candidates = get_broad_candidates(citation)
                    results[citation] = resolve_match_with_llm(citation, candidates, language)
    
    return results


# ==========================================
# 5. API Endpoints
# ==========================================

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
async def audit_document(file: UploadFile = File(...), language: str = Form("english")):
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

    extracted_data = extract_citations_with_groq(document_text, language)
    sc_citations = extracted_data["sc_cases"]
    hc_citations = extracted_data["hc_cases"]
    case_details = extracted_data["details"]

    if not sc_citations and not hc_citations:
        return JSONResponse({"message": "No citations found in the document.", "results": []})

    final_report = []

    # ── Batch-resolve all SC citations in fewer LLM calls ──
    sc_candidate_pairs = []
    for citation in sc_citations:
        candidates = get_broad_candidates(citation)
        sc_candidate_pairs.append((citation, candidates))
    
    # This sends up to 5 citations per LLM call instead of 1
    sc_verification_results = batch_resolve_matches_with_llm(sc_candidate_pairs, language)

    for citation in sc_citations:
        verification_result = sc_verification_results.get(citation, {
            "status": "ERROR", "message": "Verification not returned from batch", "confidence": 0
        })
        
        quote_verification = {}
        claim = case_details.get(citation, {}).get("attributed_claim", "")
        
        if "🟢" in verification_result.get("status", ""):
            source_file_path = verification_result.get("file_to_open", "")
            if not source_file_path or source_file_path.strip() == "":
                quote_verification = {"status": "⚠️ SKIPPED", "reason": "No PDF path available in database for this case."}
            else:
                quote_verification = verify_quotation(claim, source_file_path, language)
        else:
            quote_verification = {"status": "⚠️ SKIPPED", "reason": "Case was not verified, skipping quote check."}

        final_report.append({
            "target_citation": citation,
            "court_type": "Supreme Court / Unknown",
            "verification": verification_result,
            "quote_verification": quote_verification
        })

    # ── Batch-resolve HC citations too ──
    hc_candidate_pairs = []
    for citation in hc_citations:
        candidates = get_broad_candidates(citation)
        hc_candidate_pairs.append((citation, candidates))
    
    hc_verification_results = batch_resolve_matches_with_llm(hc_candidate_pairs, language) if hc_candidate_pairs else {}

    for citation in hc_citations:
        verification_result = hc_verification_results.get(citation, None)
        
        if verification_result and verification_result.get("status") != "🔴 NO CANDIDATES":
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
        "results": final_report,
        "language": language
    })

@app.post("/audit-multiple")
async def audit_multiple(files: List[UploadFile] = File(...), language: str = Form("english")):
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

        extracted_data = extract_citations_with_groq(document_text, language)
        sc_citations = extracted_data["sc_cases"]
        hc_citations = extracted_data["hc_cases"]
        case_details = extracted_data["details"]
        total_sc += len(sc_citations)
        total_hc += len(hc_citations)

        file_report = []
        
        # Batch-resolve SC citations for this file
        sc_candidate_pairs = [(c, get_broad_candidates(c)) for c in sc_citations]
        sc_batch_results = batch_resolve_matches_with_llm(sc_candidate_pairs, language)
        
        for citation in sc_citations:
            verification_result = sc_batch_results.get(citation, {
                "status": "ERROR", "message": "Batch verification missing", "confidence": 0
            })
            
            quote_verification = {}
            claim = case_details.get(citation, {}).get("attributed_claim", "")
            
            if "🟢" in verification_result.get("status", ""):
                source_file_path = verification_result.get("file_to_open", "")
                if source_file_path and source_file_path.strip():
                    quote_verification = verify_quotation(claim, source_file_path, language)
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
        "documents": all_results,
        "language": language
    })

@app.post("/verify-citation")
async def verify_single_citation(req: CitationRequest):
    citation = req.citation.strip()
    if not citation:
        raise HTTPException(status_code=400, detail="Citation text cannot be empty.")

    candidates = get_broad_candidates(citation)
    result = resolve_match_with_llm(citation, candidates, req.language)

    hc_keywords = ['high court', ' hc ', 'del hc', 'bom hc', 'mad hc', 'cal hc']
    court_type = "High Court" if any(k in citation.lower() for k in hc_keywords) else "Supreme Court / Unknown"

    return JSONResponse({
        "target_citation": citation,
        "court_type": court_type,
        "verification": result,
        "language": req.language
    })

@app.post("/chat")
async def legal_chat(payload: ChatMessage):
    lang_suffix = get_language_prompt_suffix(payload.language)
    
    system_prompt = f"""You are LexAI, an expert AI legal assistant specializing in Indian law, particularly Supreme Court and High Court jurisprudence.

You help lawyers, legal researchers, and students by:
- Explaining legal concepts, sections, and acts
- Summarizing cases and precedents
- Analyzing audit results from the Citation Auditor tool
- Answering questions about Indian constitutional law, IPC, CrPC, and civil procedure
- Helping identify potential legal arguments and precedents

Always be precise, cite relevant law where possible, and note when you are uncertain.
If the user shares audit results, use them as context to give tailored advice.
Keep responses concise but thorough. Format with bullet points when listing items.

{lang_suffix}"""

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
    lang_suffix = get_language_prompt_suffix(req.language)

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

Be direct and professional. Do not use markdown headers.

{lang_suffix}"""

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
            },
            "language": req.language
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")
    

@app.post("/voice-analyze")
async def voice_analyze(req: VoiceAnalyzeRequest):
    """
    Takes a transcribed legal query and returns structured:
    - advice (text)
    - citations (list of case names)
    - legal_suggestions (list of action items)
    """
    if not req.transcript or not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    lang_suffix = get_language_prompt_suffix(req.language)
    prompt = f"""You are LexAI, an expert AI legal assistant specializing in Indian law (Supreme Court, High Court, IPC, CrPC, Constitution).

A user has spoken the following legal query:
"{req.transcript}"

Analyze this query and respond ONLY with valid JSON in this exact format:
{{
  "advice": "A clear, concise 3-5 sentence legal advice paragraph addressing the user's situation under Indian law. Be specific and actionable.",
  "citations": [
    "Case Name 1 v. Respondent (Year) — one-line relevance",
    "Case Name 2 v. Respondent (Year) — one-line relevance",
    "Case Name 3 v. Respondent (Year) — one-line relevance"
  ],
  "legal_suggestions": [
    "Actionable suggestion 1",
    "Actionable suggestion 2",
    "Actionable suggestion 3",
    "Actionable suggestion 4"
  ]
}}

Rules:
- citations: provide 2-4 real, relevant Indian Supreme Court or High Court cases. If none are clearly relevant, return an empty array.
- legal_suggestions: provide 3-5 clear, numbered action steps the person should take.
- Keep advice professional, clear, and grounded in Indian law.
- Respond ONLY with the JSON. No extra text.

{lang_suffix}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are LexAI, a senior Indian legal analyst. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1200
        )
        result = json.loads(response.choices[0].message.content)
        return JSONResponse({
            "advice": result.get("advice", ""),
            "citations": result.get("citations", []),
            "legal_suggestions": result.get("legal_suggestions", [])
        })
    except Exception as e:
        logger.error(f"❌ Voice analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# ==========================================
# Case Research Endpoints

# ==========================================

@app.get("/case-research", response_class=FileResponse)
async def serve_case_research():
    return FileResponse("frontend/templates/case-research.html")


@app.get("/api/case-research/stats")
def case_research_stats():
    """Return stats about the case intelligence database."""
    if case_cards_df.empty:
        return {
            "loaded": False,
            "total_cases": 0,
            "message": "Case intelligence database not loaded. Run build_case_index.py first."
        }

    domains = {}
    if 'legal_domain' in case_cards_df.columns:
        domain_counts = case_cards_df['legal_domain'].value_counts().to_dict()
        domains = {str(k): int(v) for k, v in domain_counts.items()}

    courts = {}
    if 'court' in case_cards_df.columns:
        court_counts = case_cards_df['court'].value_counts().head(10).to_dict()
        courts = {str(k): int(v) for k, v in court_counts.items()}

    return {
        "loaded": True,
        "total_cases": len(case_cards_df),
        "embeddings_loaded": case_embeddings_matrix is not None,
        "embedding_dimensions": (
            int(case_embeddings_matrix.shape[1])
            if case_embeddings_matrix is not None
            else 0
        ),
        "domains": domains,
        "courts": courts
    }


@app.post("/api/case-research/search")
async def search_similar_cases(req: CaseResearchRequest):
    """
    Core search endpoint. Takes a lawyer's case description and finds
    similar cases from the indexed database using semantic search.
    """
    if case_cards_df.empty or case_embeddings_matrix is None:
        raise HTTPException(
            status_code=503,
            detail="Case intelligence database not loaded. Run build_case_index.py first."
        )

    if not req.case_description.strip():
        raise HTTPException(status_code=400, detail="Case description cannot be empty.")

    num_results = min(req.num_results or 5, 20)

    # Build the search query combining all inputs
    search_parts = [req.case_description.strip()]
    if req.legal_domain:
        search_parts.append(f"Legal domain: {req.legal_domain}")
    if req.key_statutes:
        search_parts.append(f"Relevant statutes: {', '.join(req.key_statutes)}")

    search_query = " ".join(search_parts)

    logger.info(f"🔍 Case research query: {search_query[:100]}...")

    # ──────────────────────────────────────────────────
    # FIX: Align case_cards and embeddings by row count
    # The parquet may have more rows than embeddings (e.g., 
    # failed extractions added after embedding was built).
    # We only search over the rows that HAVE embeddings.
    # ──────────────────────────────────────────────────
    num_embedded = case_embeddings_matrix.shape[0]
    num_cards = len(case_cards_df)

    if num_embedded != num_cards:
        logger.warning(
            f"⚠️ Shape mismatch: {num_cards} cards vs {num_embedded} embeddings. "
            f"Using first {num_embedded} cards only."
        )

    # Work with only the rows that have embeddings
    search_df = case_cards_df.iloc[:num_embedded].copy()
    search_embeddings = case_embeddings_matrix

    # Encode the query
    query_embedding = embedder.encode(
        [search_query], normalize_embeddings=True
    )

    # Compute similarity against all case embeddings
    similarities = cosine_similarity(query_embedding, search_embeddings)[0]

    # Apply domain filter if specified (boost matching domain scores)
    if req.legal_domain and 'legal_domain' in search_df.columns:
        domain_mask = (
            search_df['legal_domain']
            .str.lower()
            .eq(req.legal_domain.lower())
            .values  # Convert to numpy array
        )
        similarities = np.where(domain_mask, similarities * 1.15, similarities)

    # Apply statute filter (boost cases mentioning same statutes)
    if req.key_statutes and 'key_statutes' in search_df.columns:
        for statute in req.key_statutes:
            statute_lower = statute.lower()
            statute_mask = search_df['key_statutes'].apply(
                lambda x: any(statute_lower in str(s).lower() for s in x)
                if isinstance(x, list) else statute_lower in str(x).lower()
            ).values  # Convert to numpy array
            similarities = np.where(statute_mask, similarities * 1.10, similarities)

    # Clip similarities to [0, 1] after boosting
    similarities = np.clip(similarities, 0.0, 1.0)

    # Get top results
    top_indices = np.argsort(similarities)[-num_results:][::-1]

    results = []
    for idx in top_indices:
        idx = int(idx)
        score = float(similarities[idx])

        if score < 0.15:
            continue

        row = search_df.iloc[idx]

        # Determine relevance tier
        if score >= 0.65:
            relevance = "🟢 Highly Relevant"
        elif score >= 0.45:
            relevance = "🟡 Moderately Relevant"
        else:
            relevance = "🟠 Potentially Relevant"

        # Build key_statutes safely
        statutes = row.get('key_statutes', [])
        if isinstance(statutes, str):
            statutes = [statutes]
        elif not isinstance(statutes, list):
            statutes = []

        principles = row.get('key_principles', [])
        if isinstance(principles, str):
            principles = [principles]
        elif not isinstance(principles, list):
            principles = []

        results.append({
            "rank": len(results) + 1,
            "similarity_score": round(score, 4),
            "relevance": relevance,
            "case_title": str(row.get('case_title', 'Unknown')),
            "court": str(row.get('court', 'Unknown')),
            "legal_domain": str(row.get('legal_domain', 'Unknown')),
            "core_legal_question": str(row.get('core_legal_question', '')),
            "holding": str(row.get('holding', '')),
            "key_principles": principles,
            "key_statutes": statutes,
            "searchable_summary": str(row.get('searchable_summary', '')),
            "source_file": str(row.get('source_file', ''))
        })

    if not results:
        return JSONResponse({
            "status": "NO_RESULTS",
            "message": "No sufficiently similar cases found. Try broadening your description.",
            "results": [],
            "query_used": search_query
        })

    return JSONResponse({
        "status": "SUCCESS",
        "total_results": len(results),
        "query_used": search_query,
        "results": results
    })


@app.post("/api/case-research/read-case")
async def read_case_pdf(req: CaseDetailRequest):
    """
    Read the full PDF of a specific case and return its text content
    so a lawyer can study it in detail.
    """
    if not req.source_file or not req.source_file.strip():
        raise HTTPException(status_code=400, detail="No source file specified.")

    source_path = req.source_file.strip()
    logger.info(f"📖 Reading case PDF: {source_path}")

    # Try multiple strategies to find and read the PDF
    full_text = ""

    # Strategy 1: Direct path
    direct = Path(source_path)
    if direct.exists() and direct.suffix == '.pdf':
        full_text = _read_pdf(direct)

    # Strategy 2: Use the existing extract function
    if not full_text:
        full_text = extract_text_from_pdf_path(source_path)

    if not full_text.strip():
        raise HTTPException(
            status_code=404,
            detail=f"Could not read PDF for: {source_path}"
        )

    return JSONResponse({
        "status": "SUCCESS",
        "source_file": source_path,
        "text_length": len(full_text),
        "full_text": full_text
    })


@app.post("/api/case-research/analyze-for-argument")
async def analyze_case_for_argument(payload: dict):
    """
    Given a lawyer's case description and a specific found case,
    use LLM to analyze how the found case can strengthen the lawyer's argument.
    """
    case_description = payload.get("case_description", "")
    found_case = payload.get("found_case", {})
    case_text = payload.get("case_text", "")

    if not case_description or not found_case:
        raise HTTPException(
            status_code=400,
            detail="Both case_description and found_case are required."
        )

    # Build context from the found case
    case_context = f"""
Case Title: {found_case.get('case_title', 'Unknown')}
Court: {found_case.get('court', 'Unknown')}
Legal Domain: {found_case.get('legal_domain', 'Unknown')}
Core Legal Question: {found_case.get('core_legal_question', '')}
Holding: {found_case.get('holding', '')}
Key Principles: {', '.join(found_case.get('key_principles', []))}
Summary: {found_case.get('searchable_summary', '')}
"""

    # If we have the full text, add relevant excerpts
    text_section = ""
    if case_text:
        # Take first 3000 and last 2000 chars for the LLM
        if len(case_text) > 5000:
            text_section = (
                f"\n\nRelevant excerpts from the judgment:\n"
                f"{case_text[:3000]}\n\n[...]\n\n{case_text[-2000:]}"
            )
        else:
            text_section = f"\n\nFull judgment text:\n{case_text}"

    prompt = f"""You are a senior legal strategist helping a lawyer prepare their case.

THE LAWYER'S CURRENT CASE:
{case_description}

A SIMILAR PRECEDENT WAS FOUND:
{case_context}
{text_section}

YOUR TASK:
Analyze how this precedent can be used to STRENGTHEN the lawyer's case. Provide:

1. **Relevance Assessment**: How closely does this precedent relate to the lawyer's case? What are the key similarities?

2. **Key Arguments to Extract**: What specific legal principles, holdings, or reasoning from this case can the lawyer cite?

3. **How to Use This Case**: Concrete suggestions on how to frame arguments using this precedent. Include sample argument language.

4. **Distinguishing Factors**: Any differences between the precedent and the current case that opposing counsel might raise, and how to address them.

5. **Supporting Ratio Decidendi**: The core legal rationale that would be most persuasive if cited.

Be specific, practical, and actionable. Write as if advising a practicing lawyer."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior legal strategist specializing in Indian "
                        "jurisprudence. You help lawyers build stronger cases by "
                        "analyzing precedents and crafting compelling arguments."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        return JSONResponse({
            "status": "SUCCESS",
            "analysis": response.choices[0].message.content,
            "case_used": found_case.get('case_title', 'Unknown')
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM analysis error: {str(e)}"
        )


@app.post("/api/case-research/build-strategy")
async def build_case_strategy(payload: dict):
    """
    Given the lawyer's case and multiple similar cases found,
    build a comprehensive litigation strategy.
    """
    case_description = payload.get("case_description", "")
    similar_cases = payload.get("similar_cases", [])

    if not case_description or not similar_cases:
        raise HTTPException(
            status_code=400,
            detail="Both case_description and similar_cases are required."
        )

    # Format similar cases for the LLM
    cases_summary = ""
    for i, case in enumerate(similar_cases[:7], 1):
        cases_summary += f"""
--- Precedent {i} ---
Title: {case.get('case_title', 'Unknown')}
Court: {case.get('court', 'Unknown')}
Domain: {case.get('legal_domain', 'Unknown')}
Core Question: {case.get('core_legal_question', '')}
Holding: {case.get('holding', '')}
Principles: {', '.join(case.get('key_principles', []))}
Relevance Score: {case.get('similarity_score', 0)}
"""

    prompt = f"""You are a senior litigation strategist preparing a comprehensive case strategy.

THE LAWYER'S CASE:
{case_description}

RELEVANT PRECEDENTS FOUND ({len(similar_cases)} cases):
{cases_summary}

Generate a COMPREHENSIVE LITIGATION STRATEGY that includes:

1. **Case Strength Assessment**: Overall strength of the case based on available precedents (Strong/Moderate/Weak with reasoning)

2. **Primary Arguments** (ranked by strength):
   - For each argument, cite which precedent(s) support it
   - Include the legal principle and how it applies

3. **Chain of Precedents**: How to build a logical chain linking multiple precedents to create a compelling narrative

4. **Anticipated Counter-Arguments**: What the opposing side is likely to argue, and how to rebut using the precedents found

5. **Recommended Citation Order**: The order in which to present these precedents for maximum impact

6. **Key Phrases to Use**: Specific legal phrases from the precedents that the lawyer should quote

7. **Risk Assessment**: Any gaps in the precedent support and how to mitigate them

Be thorough, strategic, and practical. This should serve as an actionable litigation playbook."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are India's top litigation strategist. You build "
                        "winning case strategies by masterfully weaving together "
                        "precedents, statutes, and legal principles."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2500
        )

        return JSONResponse({
            "status": "SUCCESS",
            "strategy": response.choices[0].message.content,
            "precedents_used": len(similar_cases)
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Strategy generation error: {str(e)}"
        )