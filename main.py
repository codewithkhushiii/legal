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
from groq import Groq

# ==========================================
# 1. Configuration & Global State
# ==========================================
# WARNING: Hardcoding keys is okay for testing, but use env vars for production!
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
        df = df.reset_index() # Give every row a unique ID
        print(f"✅ Successfully loaded {len(df)} records into RAM!")
    else:
        print("❌ WARNING: No valid .parquet files were found. Search will fail.")
    
    yield # The API is now running and accepts requests
    
    # Optional cleanup on shutdown can go here
    print("🛑 Shutting down server...")

# Initialize FastAPI
app = FastAPI(title="Legal Citation Auditor API", lifespan=lifespan)

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
    
    # We use a dictionary to automatically deduplicate cases found in multiple chunks!
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
            
            # Add to our dictionary (this naturally prevents duplicates)
            for item in extracted_in_chunk:
                name = item.get("case_name")
                if name and name not in all_extracted_cases:
                    all_extracted_cases[name] = item
                    
        except Exception as e:
            print(f"❌ Error extracting citations from chunk {i+1}: {e}")

    # ---------------------------------------------------------
    # THE PYTHON FILTER (Runs after all chunks are processed)
    # ---------------------------------------------------------
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
    
    # Return BOTH lists now!
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
        return {"status": "🔴 NO CANDIDATES", "message": "No similar cases found in database."}
        
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
    Respond ONLY with JSON: {{"matched_id": "id_or_null", "reason": "Short reason"}}
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
        
        if matched_id and matched_id != "null":
            winning_row = df[df['index'] == int(matched_id)].iloc[0]
            return {
                "status": "🟢 VERIFIED BY AI",
                "matched_name": winning_row['title'],
                "file_to_open": winning_row.get('path', 'Unknown PDF'),
                "reason": reason
            }
        return {
            "status": "🔴 HALLUCINATION DETECTED",
            "message": f"Reason: {reason}"
        }
    except Exception as e:
        return {"status": "ERROR", "message": f"Groq API Error: {str(e)}"}

# ==========================================
# 4. API Endpoints
# ==========================================
@app.get("/")
def read_root():
    return {"message": "Legal Citation Auditor API is running! Check /docs to test it."}

@app.post("/audit-document")
async def audit_document(file: UploadFile = File(...)):
    """Upload a PDF, extract text, find citations, and verify them against the database."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # 1. Read the PDF in memory
    try:
        file_bytes = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        document_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

    if not document_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    # 2. Extract Citations (Now returns a dictionary)
    extracted_data = extract_citations_with_groq(document_text)
    sc_citations = extracted_data["sc_cases"]
    hc_citations = extracted_data["hc_cases"]
    
    if not sc_citations and not hc_citations:
        return JSONResponse({"message": "No citations found in the document.", "results": []})

    final_report = []

    # 3a. Verify the Supreme Court citations against your Parquet database
    for citation in sc_citations:
        candidates = get_broad_candidates(citation)
        verification_result = resolve_match_with_llm(citation, candidates)
        
        final_report.append({
            "target_citation": citation,
            "court_type": "Supreme Court / Unknown",
            "verification": verification_result
        })

    # 3b. Add the High Court citations to the report (Bypass the database)
    for citation in hc_citations:
        final_report.append({
            "target_citation": citation,
            "court_type": "High Court",
            "verification": {
                "status": "⚠️ SKIPPED",
                "message": "Identified as a High Court case. Not verified against the Supreme Court registry."
            }
        })

    return JSONResponse({
        "total_citations_found": len(sc_citations) + len(hc_citations),
        "supreme_court_count": len(sc_citations),
        "high_court_count": len(hc_citations),
        "results": final_report
    })