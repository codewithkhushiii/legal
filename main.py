import os
import re
import json
import io
import pandas as pd
import PyPDF2
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq

# ==========================================
# 1. Configuration & Global State
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

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
            print(f"⚠️ Could not load {file_path}: {e}")

    if all_dataframes:
        df = pd.concat(all_dataframes, ignore_index=True)
        df = df.reset_index()
        print(f"✅ Successfully loaded {len(df)} records into RAM!")
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


def extract_citations_with_groq(full_text):
    print("🧠 Splitting document into manageable chunks...")
    chunks = chunk_text(full_text)
    print(f"✂️ Created {len(chunks)} chunks for processing.")

    all_extracted_cases = {}

    for i, chunk in enumerate(chunks):
        print(f"🔍 Analyzing Chunk {i+1}/{len(chunks)}...")

        prompt = f"""
        You are an expert legal AI auditor. Read this section of a legal document.
        Extract EVERY single legal case, precedent, or judgment mentioned.
        
        For each case, classify the court based on context:
        - If it mentions "Del", "Bom", "Mad", or "High Court", classify as "High Court".
        - If it mentions "INSC", "SCC", or implies the apex court, classify as "Supreme Court".
        - If you cannot tell, classify as "Unknown".
        
        Respond ONLY with a valid JSON object in this exact format:
        {{
          "citations": [
            {{
              "case_name": "Exact Name of Case",
              "court_type": "Supreme Court",
              "reasoning": "Mentioned 'INSC' in the citation"
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
            print(f"❌ Error extracting citations from chunk {i+1}: {e}")

    supreme_court_cases = []
    high_court_cases = []

    print("\n--- 📊 Final Extraction Report (Across All Chunks) ---")
    for name, item in all_extracted_cases.items():
        court = item.get("court_type")
        reason = item.get("reasoning")

        if court == "High Court":
            print(f"📌 Flagged High Court Case: {name} (Reason: {reason})")
            high_court_cases.append(name)
        else:
            print(f"✅ Kept SC/Unknown Case: {name} (Reason: {reason})")
            supreme_court_cases.append(name)

    print("------------------------------------------------------\n")

    return {
        "sc_cases": supreme_court_cases,
        "hc_cases": high_court_cases
    }


def get_broad_candidates(case_name, max_results=15):
    if df.empty:
        return pd.DataFrame()

    cleaned = re.sub(r'[^a-zA-Z\s]', '', case_name)
    words = cleaned.split()
    stop_words = ['the', 'state', 'union', 'of', 'india', 'vs', 'v', 'versus', 'ors', 'and', 'others', 'anr']
    meaningful_words = [w for w in words if w.lower() not in stop_words]
    search_word = meaningful_words[0] if meaningful_words else (words[0] if words else "")

    if not search_word:
        return pd.DataFrame()

    matches = df[df['title'].str.contains(search_word, case=False, na=False)]
    return matches.head(max_results)


def resolve_match_with_llm(target_citation, candidates_df):
    if candidates_df.empty:
        return {"status": "🔴 NO CANDIDATES", "message": "No similar cases found in database.", "confidence": 0}

    candidate_dict = {str(row['index']): f"{row['title']} (Year: {row.get('year', 'Unknown')})"
                      for _, row in candidates_df.iterrows()}

    prompt = f"""
    You are a STRICT legal AI auditor. Verify if a Target Citation exists in the Database Candidates.
    Target Citation: "{target_citation}"
    Database Candidates: {json.dumps(candidate_dict)}
    
    RULES:
    1. Minor formatting differences ("v." vs "versus") or missing articles ("The") are ALLOWED.
    2. DIFFERENT PROPER NOUNS ARE STRICTLY PROHIBITED.
    If there is an EXACT match, return its ID. If NO EXACT MATCH, return "null".
    Also provide a confidence score 0-100 representing how certain you are.
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
            winning_row = df[df['index'] == int(matched_id)].iloc[0]
            return {
                "status": "🟢 VERIFIED BY AI",
                "matched_name": winning_row['title'],
                "file_to_open": winning_row.get('path', 'Unknown PDF'),
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
    """Returns statistics about the loaded database."""
    if df.empty:
        return {"loaded": False, "record_count": 0}
    return {
        "loaded": True,
        "record_count": len(df),
        "columns": list(df.columns)
    }


@app.post("/audit-document")
async def audit_document(file: UploadFile = File(...)):
    """Upload a PDF, extract text, find citations, and verify them against the database."""
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

    if not sc_citations and not hc_citations:
        return JSONResponse({"message": "No citations found in the document.", "results": []})

    final_report = []

    for citation in sc_citations:
        candidates = get_broad_candidates(citation)
        verification_result = resolve_match_with_llm(citation, candidates)

        final_report.append({
            "target_citation": citation,
            "court_type": "Supreme Court / Unknown",
            "verification": verification_result
        })

    for citation in hc_citations:
        # Try HC database lookup
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
            "verification": verification_result
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
    """Upload multiple PDFs and get a combined audit report."""
    all_results = []
    total_sc = 0
    total_hc = 0

    for file in files:
        if not file.filename.endswith('.pdf'):
            all_results.append({
                "filename": file.filename,
                "error": "Not a PDF file",
                "results": []
            })
            continue

        try:
            file_bytes = await file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            document_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
        except Exception as e:
            all_results.append({
                "filename": file.filename,
                "error": f"Error reading PDF: {str(e)}",
                "results": []
            })
            continue

        if not document_text.strip():
            all_results.append({
                "filename": file.filename,
                "error": "Could not extract text from PDF",
                "results": []
            })
            continue

        extracted_data = extract_citations_with_groq(document_text)
        sc_citations = extracted_data["sc_cases"]
        hc_citations = extracted_data["hc_cases"]
        total_sc += len(sc_citations)
        total_hc += len(hc_citations)

        file_report = []
        for citation in sc_citations:
            candidates = get_broad_candidates(citation)
            verification_result = resolve_match_with_llm(citation, candidates)
            file_report.append({
                "target_citation": citation,
                "court_type": "Supreme Court / Unknown",
                "verification": verification_result
            })

        for citation in hc_citations:
            file_report.append({
                "target_citation": citation,
                "court_type": "High Court",
                "verification": {
                    "status": "⚠️ SKIPPED",
                    "message": "High Court case bypassed.",
                    "confidence": 0
                }
            })

        all_results.append({
            "filename": file.filename,
            "citations_found": len(sc_citations) + len(hc_citations),
            "sc_count": len(sc_citations),
            "hc_count": len(hc_citations),
            "results": file_report
        })

    verified = sum(
        1 for doc in all_results
        for r in doc.get("results", [])
        if "🟢" in (r.get("verification") or {}).get("status", "")
    )
    fabricated = sum(
        1 for doc in all_results
        for r in doc.get("results", [])
        if "HALLUCINATION" in (r.get("verification") or {}).get("status", "")
    )

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
    """Manually verify a single citation by name."""
    citation = req.citation.strip()
    if not citation:
        raise HTTPException(status_code=400, detail="Citation text cannot be empty.")

    candidates = get_broad_candidates(citation)
    result = resolve_match_with_llm(citation, candidates)

    # Classify court type from text
    hc_keywords = ['high court', ' hc ', 'del hc', 'bom hc', 'mad hc', 'cal hc']
    court_type = "High Court" if any(k in citation.lower() for k in hc_keywords) else "Supreme Court / Unknown"

    return JSONResponse({
        "target_citation": citation,
        "court_type": court_type,
        "verification": result
    })


@app.post("/chat")
async def legal_chat(payload: ChatMessage):
    """Legal AI Chatbot powered by Groq."""
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

    # Add audit context if provided
    if payload.audit_context:
        messages.append({
            "role": "system",
            "content": f"Current audit context (results from the last citation audit):\n{payload.audit_context}"
        })

    # Add conversation history
    if payload.history:
        for msg in payload.history[-10:]:  # Keep last 10 messages to avoid context overflow
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": payload.message})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        reply = response.choices[0].message.content
        return JSONResponse({"reply": reply})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")


@app.post("/summarize")
async def generate_summary(req: SummaryRequest):
    """Generate a plain-language audit summary."""
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
        summary = response.choices[0].message.content
        risk = "High" if len(fabricated) > 2 else ("Medium" if len(fabricated) > 0 else "Low")
        return JSONResponse({
            "summary": summary,
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