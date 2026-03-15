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

# ==========================================
# 1. Configuration & Global State
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

print("📥 Loading Embedding Model for RAG...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Global variable to hold our database
df = pd.DataFrame()

# ==========================================
# 2. Server Startup (Load Data)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df
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

    yield
    print("🛑 Shutting down server...")

# Initialize FastAPI
app = FastAPI(title="Legal Citation Auditor API", lifespan=lifespan)

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