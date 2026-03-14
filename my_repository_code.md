
---

### File: `main.py`

```py
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
import re
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


print("📥 Loading Embedding Model for RAG...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')


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
def extract_text_from_source(db_path_value: str) -> str:
    """Dynamically finds and reads the PDF based on the nested year-based folder structure."""
    if not db_path_value or pd.isna(db_path_value):
        return ""
        
    file_id = str(db_path_value).strip()
    
    # 1. Extract the year from the file ID (e.g., '2024' from '2024_1_1_10')
    year_match = re.match(r'^(\d{4})', file_id)
    if not year_match:
        print(f"⚠️ Could not determine year from file ID: {file_id}")
        return ""
        
    year = year_match.group(1)
    
    # 2. Build the expected folder paths (handling both 'judgement' and 'judgment' spellings)
    possible_base_folders = [f"judgement_{year}", f"judgment_{year}"]
    sub_folder = f"extracted_{year}_case"
    
    file_path = None
    
    # Check the exact locations first (This is lightning fast ⚡)
    for base in possible_base_folders:
        candidate_path = Path(base) / sub_folder / f"{file_id}.pdf"
        if candidate_path.exists():
            file_path = candidate_path
            break
            
    # 3. Fallback: If the exact folder isn't found, do a quick global search
    if not file_path:
        print(f"⚠️ Strict path not found for {file_id}. Searching globally...")
        found_files = list(Path('.').rglob(f"{file_id}.pdf"))
        if found_files:
            # Drop it if it's in a virtual environment folder
            valid_files = [f for f in found_files if 'venv' not in f.parts]
            if valid_files:
                file_path = valid_files[0]
            
    if not file_path or not file_path.exists():
        print(f"❌ File completely missing from disk: {file_id}.pdf")
        return ""

    print(f"📖 Successfully located and reading PDF: {file_path}")
    
    # 4. Read the PDF
    try:
        reader = PyPDF2.PdfReader(str(file_path))
        return "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"⚠️ Error reading PDF {file_path}: {e}")
        return ""
def verify_quotation(attributed_claim: str, source_file_path: str):
    """The RAG Engine: Finds the context in the judgment and uses LLM to verify the claim."""
    if not attributed_claim:
        return {"status": "⚠️ SKIPPED", "reason": "No specific claim was extracted to verify."}

    print(f"📖 Reading source file for RAG: {source_file_path}")
    source_text = extract_text_from_source(source_file_path)
    
    if not source_text.strip():
        return {"status": "⚠️ ERROR", "reason": "Could not extract text from the source judgment."}

    # 1. Chunking: Try to split by paragraph first
    paragraphs = [p.strip() for p in source_text.split('\n\n') if len(p.strip()) > 100]

    # Fallback: If the PDF was poorly formatted and didn't have double newlines, chunk manually
    if len(paragraphs) < 5:
        paragraphs = chunk_text(source_text, chunk_size=1200, overlap=200)

    if not paragraphs:
        return {"status": "⚠️ ERROR", "reason": "No valid text chunks found in source."}

    # 2. Embedding generation
    print(f"🧠 Embedding {len(paragraphs)} chunks to find the quote...")
    claim_embedding = embedder.encode([attributed_claim])
    para_embeddings = embedder.encode(paragraphs)

    # 3. Semantic Search via Cosine Similarity
    similarities = cosine_similarity(claim_embedding, para_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_paragraph = paragraphs[best_match_idx]
    max_score = similarities[best_match_idx]

    # If the score is incredibly low, the lawyer entirely hallucinated the quote
    if max_score < 0.25:
         return {
             "status": "🔴 FABRICATED QUOTE", 
             "reason": f"No matching concepts found in the judgment. (Max similarity: {max_score:.2f})",
             "closest_text_found": best_paragraph[:200] + "..."
         }

    # 4. LLM Natural Language Inference (NLI)
    # The embeddings found the right neighborhood; now Groq determines if it actually matches.
    verification_prompt = f"""
    You are a strict legal auditor. 
    A lawyer claims this case states: "{attributed_claim}"
    
    The most relevant paragraph found in the actual case file is:
    "{best_paragraph}"
    
    Does the actual paragraph SUPPORT, CONTRADICT, or is it UNSUPPORTED/UNRELATED to the claim?
    Respond ONLY in JSON format:
    {{"verdict": "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED", "explanation": "Brief reasoning"}}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": verification_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        
        status_map = {
            "SUPPORTED": "🟢 QUOTE VERIFIED",
            "CONTRADICTED": "🔴 QUOTE CONTRADICTED",
            "UNSUPPORTED": "🟠 QUOTE UNSUPPORTED"
        }
        
        return {
            "status": status_map.get(result.get("verdict"), "⚠️ UNKNOWN"),
            "reason": result.get("explanation"),
            "found_paragraph": best_paragraph
        }
    except Exception as e:
         return {"status": "⚠️ ERROR", "reason": f"LLM Error: {str(e)}"}

async def extract_from_chunk_async(chunk: str):
    prompt = f"""
    You are an expert legal AI auditor. Extract EVERY legal case mentioned in this text.
    Also extract the exact quote, claim, or legal principle attributed to it.
    
    Respond ONLY with valid JSON:
    {{
      "citations": [
        {{
          "case_name": "Name v. Name",
          "court_type": "Supreme Court",
          "attributed_claim": "The court held that..."
        }}
      ]
    }}
    
    Text:
    {chunk}
    """
    
    # Because Groq is fast, making parallel async calls here will process 
    # a 50 page document in the time it takes to read 2 pages.
    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content).get("citations", [])
    except Exception as e:
        print(f"Error on chunk: {e}")
        return []

async def process_full_petition(pdf_reader):
    print("🧠 Filtering 50-page petition for relevant citation neighborhoods...")
    
    # 1. Slide through the PDF and drop useless pages
    relevant_chunks = []
    for i in range(0, len(pdf_reader.pages), 2): # Step by 2
        text = (pdf_reader.pages[i].extract_text() or "")
        if i + 1 < len(pdf_reader.pages):
            text += "\n" + (pdf_reader.pages[i+1].extract_text() or "")
            
        if contains_citation_markers(text):
            relevant_chunks.append(text)
            
    print(f"✂️ Reduced petition down to {len(relevant_chunks)} critical chunks.")

    # 2. Process all relevant chunks in parallel
    tasks = [extract_from_chunk_async(chunk) for chunk in relevant_chunks]
    results = await asyncio.gather(*tasks)
    
    # 3. Flatten the results list
    all_extracted_cases = []
    for result_list in results:
        all_extracted_cases.extend(result_list)
        
    # Deduplicate by case name just in case the same case was mentioned across chunks
    unique_cases = {item['case_name']: item for item in all_extracted_cases if item.get('case_name')}
    
    return list(unique_cases.values())

def extract_and_filter_pdf(pdf_reader) -> list:
    """Extracts text page by page and only keeps chunks that look like they contain citations."""
    relevant_chunks = []
    
    # Process with a sliding window of 2 pages to prevent cutting citations in half across pages
    for i in range(len(pdf_reader.pages)):
        current_page = pdf_reader.pages[i].extract_text() or ""
        next_page = pdf_reader.pages[i+1].extract_text() or "" if i + 1 < len(pdf_reader.pages) else ""
        
        combined_text = current_page + "\n" + next_page
        
        if contains_citation_markers(combined_text):
            relevant_chunks.append(combined_text)
            
    # Deduplicate chunks if necessary (since we used a sliding window)
    # For a quick implementation, we can just step by 2:
    
    return relevant_chunks
def contains_citation_markers(text_chunk: str) -> bool:
    """Fast regex check to see if a chunk might contain a legal case."""
    # [vV]s?\.?\s catches: v. | v | vs. | vs | V. | V | VS. | VS
    # re.IGNORECASE ensures SCC catches scc, Scc, etc.
    pattern = r'([vV]s?\.?\s|versus|AIR\s\d{4}|SCC|\d{4}\s\d+\sSCC|High Court|Supreme Court)'
    
    return bool(re.search(pattern, text_chunk, re.IGNORECASE))


def get_broad_candidates(case_name, max_results=15):
    if df.empty:
        return pd.DataFrame()

    query = str(case_name).strip()
    candidates = pd.DataFrame()

    # --- STRATEGY 1: NUMERIC MATCHING (Appeal Nos, Citations, Diary Nos) ---
    # Extract all numbers from the citation (e.g., '3030' and '2022' from "Civil Appeal 3030/2022")
    nums = re.findall(r'\d+', query)
    
    if len(nums) >= 2:
        # If there are at least 2 numbers, use AND logic (must contain both numbers somewhere)
        mask = pd.Series(True, index=df.index)
        for num in nums[:3]:  # Limit to top 3 numbers to avoid over-filtering
            col_mask = (
                df['title'].astype(str).str.contains(num, na=False) |
                df['description'].astype(str).str.contains(num, na=False) |
                df['case_id'].astype(str).str.contains(num, na=False) |
                df['nc_display'].astype(str).str.contains(num, na=False) |
                df['citation'].astype(str).str.contains(num, na=False)
            )
            mask = mask & col_mask
        
        num_matches = df[mask]
        if not num_matches.empty:
            candidates = pd.concat([candidates, num_matches])

    # --- STRATEGY 2: PARTY NAME MATCHING (Text Search) ---
    # Remove numbers and special characters to isolate names
    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', query)
    words = cleaned_text.split()
    
    # Expanded stop words list specific to legal citations
    stop_words = {'the', 'state', 'union', 'of', 'india', 'vs', 'v', 'versus', 'ors', 'and', 'others', 'anr', 
                  'supra', 'ltd', 'limited', 'civil', 'appeal', 'special', 'leave', 'petition', 'diary', 'no', 'honble'}
    
    # Get meaningful words and SORT THEM BY LENGTH
    # Why? Longest words are usually unique names (e.g., "Pushpam" instead of "Mary"), which beats spelling errors.
    meaningful_words = [w for w in words if len(w) > 2 and w.lower() not in stop_words]
    meaningful_words.sort(key=len, reverse=True)

    if meaningful_words:
        mask = pd.Series(True, index=df.index)
        # Require the top 2 longest words to both be present (AND logic)
        for word in meaningful_words[:2]:
            col_mask = (
                df['title'].astype(str).str.contains(word, case=False, na=False) |
                df['petitioner'].astype(str).str.contains(word, case=False, na=False) |
                df['respondent'].astype(str).str.contains(word, case=False, na=False)
            )
            mask = mask & col_mask
        
        text_matches = df[mask]
        if not text_matches.empty:
            candidates = pd.concat([candidates, text_matches])
            
    # --- STRATEGY 3: FALLBACK (Single Keyword) ---
    # If strict AND logic failed, fallback to searching just the single longest unique word
    if candidates.empty and meaningful_words:
        longest_word = meaningful_words[0]
        mask = (
            df['title'].astype(str).str.contains(longest_word, case=False, na=False) |
            df['petitioner'].astype(str).str.contains(longest_word, case=False, na=False) |
            df['respondent'].astype(str).str.contains(longest_word, case=False, na=False)
        )
        fallback_matches = df[mask]
        candidates = pd.concat([candidates, fallback_matches])

    # Drop duplicates and return the top N matches to the LLM
    if not candidates.empty:
        candidates = candidates.drop_duplicates(subset=['index'])
        return candidates.head(max_results)

    return pd.DataFrame()

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
import re
import asyncio

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


async def extract_from_chunk_async(chunk: str):
    prompt = f"""
    You are an expert legal AI auditor. Extract EVERY legal case mentioned in this text.
    Also extract the exact quote, claim, or legal principle attributed to it.
    
    Respond ONLY with valid JSON:
    {{
      "citations": [
        {{
          "case_name": "Name v. Name",
          "court_type": "Supreme Court",
          "attributed_claim": "The court held that..."
        }}
      ]
    }}
    
    Text:
    {chunk}
    """
    
    # Because Groq is fast, making parallel async calls here will process 
    # a 50 page document in the time it takes to read 2 pages.
    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content).get("citations", [])
    except Exception as e:
        print(f"Error on chunk: {e}")
        return []

async def process_full_petition(pdf_reader):
    print("🧠 Filtering 50-page petition for relevant citation neighborhoods...")
    
    # 1. Slide through the PDF and drop useless pages
    relevant_chunks = []
    for i in range(0, len(pdf_reader.pages), 2): # Step by 2
        text = (pdf_reader.pages[i].extract_text() or "")
        if i + 1 < len(pdf_reader.pages):
            text += "\n" + (pdf_reader.pages[i+1].extract_text() or "")
            
        if contains_citation_markers(text):
            relevant_chunks.append(text)
            
    print(f"✂️ Reduced petition down to {len(relevant_chunks)} critical chunks.")

    # 2. Process all relevant chunks in parallel
    tasks = [extract_from_chunk_async(chunk) for chunk in relevant_chunks]
    results = await asyncio.gather(*tasks)
    
    # 3. Flatten the results list
    all_extracted_cases = []
    for result_list in results:
        all_extracted_cases.extend(result_list)
        
    # Deduplicate by case name just in case the same case was mentioned across chunks
    unique_cases = {item['case_name']: item for item in all_extracted_cases if item.get('case_name')}
    
    return list(unique_cases.values())

def extract_and_filter_pdf(pdf_reader) -> list:
    """Extracts text page by page and only keeps chunks that look like they contain citations."""
    relevant_chunks = []
    
    # Process with a sliding window of 2 pages to prevent cutting citations in half across pages
    for i in range(len(pdf_reader.pages)):
        current_page = pdf_reader.pages[i].extract_text() or ""
        next_page = pdf_reader.pages[i+1].extract_text() or "" if i + 1 < len(pdf_reader.pages) else ""
        
        combined_text = current_page + "\n" + next_page
        
        if contains_citation_markers(combined_text):
            relevant_chunks.append(combined_text)
            
    # Deduplicate chunks if necessary (since we used a sliding window)
    # For a quick implementation, we can just step by 2:
    
    return relevant_chunks
def contains_citation_markers(text_chunk: str) -> bool:
    """Fast regex check to see if a chunk might contain a legal case."""
    # [vV]s?\.?\s catches: v. | v | vs. | vs | V. | V | VS. | VS
    # re.IGNORECASE ensures SCC catches scc, Scc, etc.
    pattern = r'([vV]s?\.?\s|versus|AIR\s\d{4}|SCC|\d{4}\s\d+\sSCC|High Court|Supreme Court)'
    
    return bool(re.search(pattern, text_chunk, re.IGNORECASE))

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
    
        # 🧠 4. QUOTE VERIFICATION (RAG)
        quote_verification = {}
        if "🟢" in verification_result.get("status", ""):
             # Case exists! Now let's check the quote.
             source_file_path = verification_result.get("file_to_open")
             
             # Check if we actually have a valid file path in the DB
             if pd.isna(source_file_path) or not str(source_file_path).strip():
                 quote_verification = {"status": "⚠️ SKIPPED", "reason": "No PDF path available in database for this case."}
             else:
                 quote_verification = verify_quotation(claim, str(source_file_path))
        else:
             quote_verification = {"status": "⚠️ SKIPPED", "reason": "Case was not verified, skipping quote check."}

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
```

---

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

### File: `frontend/styles.css`

```css
/* ==========================================
   LEGAL CITATION AUDITOR
   Supreme Court Themed UI
   Updated: HC/SC court classification support
   ========================================== */

:root {
    /* Court Colors */
    --parchment:        #0d0d0d;
    --parchment-light:  #141414;
    --parchment-mid:    #1a1a1a;
    --mahogany:         #1e1410;
    --mahogany-light:   #2a1c14;

    /* Gold / Brass */
    --gold:             #c9a84c;
    --gold-bright:      #e8c860;
    --gold-dim:         #c9a84c33;
    --gold-glow:        #c9a84c55;
    --brass:            #b8943f;

    /* Seal / Accent Colors */
    --seal-green:       #2d6a4f;
    --seal-green-dim:   #2d6a4f44;
    --seal-green-bright:#40916c;
    --verdict-red:      #a4161a;
    --verdict-red-dim:  #a4161a44;
    --verdict-red-bright:#d00000;
    --amber:            #e09f3e;
    --amber-dim:        #e09f3e44;
    --royal-blue:       #023e8a;
    --royal-blue-dim:   #023e8a44;
    --royal-blue-bright:#0077b6;
    --maroon:           #800020;

    /* Neutrals */
    --text-primary:     #d4c5a9;
    --text-secondary:   #9a8c73;
    --text-dim:         #5c5241;
    --text-bright:      #f0e6d0;
    --text-heading:     #e8d5a3;

    /* Borders */
    --border:           #2a2419;
    --border-gold:      #c9a84c22;

    /* Misc */
    --radius:           8px;
    --radius-sm:        5px;
    --glass:            rgba(13, 13, 13, 0.92);
    --transition:       all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --font-serif:       'Playfair Display', 'Cormorant Garamond', Georgia, serif;
    --font-body:        'Cormorant Garamond', Georgia, serif;
    --font-mono:        'JetBrains Mono', 'Courier New', monospace;
    --font-ui:          'Inter', -apple-system, sans-serif;
}
breakdown.hidden { display: none !important; }

.cb-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: var(--gold);
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}

.cb-title i {
    font-size: 0.75rem;
}

.cb-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}

.cb-row:last-child { margin-bottom: 0; }

.cb-bar-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 1px;
    color: var(--text-secondary);
    min-width: 100px;
    text-align: right;
}

.cb-bar-track {
    flex: 1;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
}

.cb-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    min-width: 0;
}

.cb-bar-fill.sc-fill {
    background: linear-gradient(90deg, var(--gold), var(--gold-bright));
    box-shadow: 0 0 8px var(--gold-dim);
}

.cb-bar-fill.hc-fill {
    background: linear-gradient(90deg, var(--royal-blue), var(--royal-blue-bright));
    box-shadow: 0 0 8px var(--royal-blue-dim);
}

.cb-bar-count {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-bright);
    min-width: 24px;
    text-align: center;
}

/* ==========================================
   CHAMBER IDLE STATE (Scales)
   ========================================== */
.chamber-idle {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
}

.idle-scales-container {
    width: 200px;
    height: 160px;
    position: relative;
    margin-bottom: 2rem;
}

.scales-beam {
    position: relative;
    width: 100%;
    height: 100%;
}

.beam-line {
    position: absolute;
    top: 30px;
    left: 20px;
    right: 20px;
    height: 3px;
    background: var(--gold);
    border-radius: 3px;
    box-shadow: 0 0 10px var(--gold-dim);
}

.scales-pivot {
    position: absolute;
    top: 8px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 2.5rem;
    filter: drop-shadow(0 0 10px var(--gold-dim));
}

.scale-pan {
    position: absolute;
    top: 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.left-pan  { left: 10px;  animation: panSwing 4s ease-in-out infinite; }
.right-pan { right: 10px; animation: panSwing 4s ease-in-out infinite reverse; }

@keyframes panSwing {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(10px); }
}

.pan-chains {
    display: flex;
    gap: 12px;
    margin-bottom: 3px;
}

.chain {
    width: 1px;
    height: 35px;
    background: repeating-linear-gradient(
        to bottom,
        var(--gold) 0px,
        var(--gold) 3px,
        transparent 3px,
        transparent 6px
    );
}

.pan-dish {
    width: 55px;
    height: 8px;
    background: var(--gold-dim);
    border-radius: 0 0 50% 50%;
    border: 1px solid var(--gold-dim);
}

.pan-label {
    margin-top: 6px;
    font-family: var(--font-mono);
    font-size: 0.55rem;
    letter-spacing: 2px;
    color: var(--text-dim);
}

.chamber-idle h3 {
    font-family: var(--font-serif);
    font-size: 1.1rem;
    letter-spacing: 4px;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.chamber-idle p {
    font-size: 0.85rem;
    color: var(--text-dim);
    max-width: 350px;
    text-align: center;
    line-height: 1.6;
    font-style: italic;
}

/* ==========================================
   CHAMBER DELIBERATION STATE
   ========================================== */
.chamber-deliberation {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 4rem 2rem;
    animation: fadeIn 0.5s ease;
}

.deliberation-visual {
    width: 100px;
    height: 100px;
    position: relative;
    margin-bottom: 2rem;
}

.quill-body {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 3rem;
    color: var(--gold);
    animation: quillWrite 2s ease-in-out infinite;
    filter: drop-shadow(0 0 15px var(--gold-dim));
}

@keyframes quillWrite {
    0%, 100% { transform: translate(-50%, -50%) rotate(-5deg); }
    50% { transform: translate(-50%, -50%) rotate(5deg) translateX(5px); }
}

.ink-drops {
    position: absolute;
    bottom: 5px;
    left: 50%;
    transform: translateX(-50%);
}

.ink-drop {
    display: inline-block;
    width: 4px;
    height: 4px;
    background: var(--gold);
    border-radius: 50%;
    margin: 0 3px;
    opacity: 0;
    animation: inkDrip 1.5s ease infinite;
    animation-delay: calc(var(--d) * 0.3s);
}

@keyframes inkDrip {
    0%   { opacity: 0; transform: translateY(-10px); }
    50%  { opacity: 1; }
    100% { opacity: 0; transform: translateY(10px); }
}

.delib-title {
    font-family: var(--font-serif);
    font-size: 1.15rem;
    letter-spacing: 4px;
    color: var(--gold);
    margin-bottom: 0.5rem;
}

.delib-sub {
    font-size: 0.8rem;
    color: var(--text-dim);
    font-style: italic;
    margin-bottom: 2rem;
}

.delib-steps {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    width: 100%;
    max-width: 360px;
}

.d-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: var(--text-dim);
    transition: var(--transition);
    padding: 0.4rem 0;
}

.d-step.active { color: var(--gold); }
.d-step.done   { color: var(--seal-green-bright); }

.d-step-marker {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1.5px solid var(--text-dim);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-serif);
    font-size: 0.7rem;
    font-weight: 700;
    transition: var(--transition);
    flex-shrink: 0;
}

.d-step.active .d-step-marker {
    border-color: var(--gold);
    color: var(--gold);
    box-shadow: 0 0 12px var(--gold-dim);
    animation: lampPulse 1.5s ease infinite;
}

.d-step.done .d-step-marker {
    border-color: var(--seal-green-bright);
    background: var(--seal-green-dim);
    color: var(--seal-green-bright);
}

/* ==========================================
   RESULTS / JUDGMENT ROLL
   ========================================== */
.chamber-results {
    animation: fadeIn 0.5s ease;
}

.judgment-filters {
    display: flex;
    gap: 0.4rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.jf-tab {
    padding: 0.4rem 0.9rem;
    border-radius: 100px;
    border: 1px solid var(--border);
    background: var(--parchment-mid);
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 1px;
    cursor: pointer;
    transition: var(--transition);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.jf-tab:hover {
    border-color: var(--gold-dim);
    color: var(--text-bright);
}

.jf-tab.active {
    background: var(--gold);
    border-color: var(--gold);
    color: var(--parchment);
}

.jf-count {
    background: rgba(0,0,0,0.2);
    padding: 0.1rem 0.35rem;
    border-radius: 100px;
    font-size: 0.55rem;
    font-weight: 700;
}

.jf-tab.active .jf-count {
    background: rgba(0,0,0,0.3);
}

.judgment-roll {
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
}

/* Individual Judgment Card */
.j-card {
    background: var(--parchment-light);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    cursor: pointer;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
    animation: cardReveal 0.5s ease backwards;
}

.j-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    width: 4px;
}

.j-card.verified::before     { background: var(--seal-green-bright); }
.j-card.hallucinated::before { background: var(--verdict-red-bright); }
.j-card.skipped::before      { background: var(--royal-blue-bright); }
.j-card.no-match::before     { background: var(--amber); }

.j-card:hover {
    border-color: var(--gold-dim);
    transform: translateX(4px);
    box-shadow: 0 4px 25px rgba(0,0,0,0.4);
}

@keyframes cardReveal {
    from { opacity: 0; transform: translateY(15px); }
    to   { opacity: 1; transform: translateY(0); }
}

.j-card-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.j-case-name {
    font-family: var(--font-serif);
    font-weight: 600;
    font-size: 1rem;
    color: var(--text-bright);
    line-height: 1.4;
    flex: 1;
}

.j-verdict-badge {
    padding: 0.25rem 0.7rem;
    border-radius: 100px;
    font-family: var(--font-mono);
    font-size: 0.55rem;
    letter-spacing: 1.5px;
    font-weight: 700;
    white-space: nowrap;
    flex-shrink: 0;
}

.j-card.verified .j-verdict-badge {
    background: var(--seal-green-dim);
    color: var(--seal-green-bright);
    border: 1px solid var(--seal-green-dim);
}

.j-card.hallucinated .j-verdict-badge {
    background: var(--verdict-red-dim);
    color: var(--verdict-red-bright);
    border: 1px solid var(--verdict-red-dim);
}

.j-card.skipped .j-verdict-badge {
    background: var(--royal-blue-dim);
    color: var(--royal-blue-bright);
    border: 1px solid var(--royal-blue-dim);
}

.j-card.no-match .j-verdict-badge {
    background: var(--amber-dim);
    color: var(--amber);
    border: 1px solid var(--amber-dim);
}

.j-card-details {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}

.j-detail {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
}

.j-detail i {
    width: 14px;
    text-align: center;
    color: var(--text-dim);
    font-size: 0.7rem;
}

.j-detail .j-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 1px;
    color: var(--text-dim);
    min-width: 55px;
}

.j-detail .j-value {
    color: var(--text-secondary);
    font-style: italic;
}

.j-card-foot {
    margin-top: 0.75rem;
    padding-top: 0.65rem;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.j-read-order {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 1.5px;
    color: var(--gold);
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.j-serial {
    font-family: var(--font-mono);
    font-size: 0.55rem;
    color: var(--text-dim);
    letter-spacing: 1px;
}

/* ==========================================
   ORDER MODAL
   ========================================== */
.order-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.75);
    backdrop-filter: blur(6px);
    z-index: 9000;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeIn 0.2s ease;
}

.order-sheet {
    background: var(--parchment-light);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    width: 92%;
    max-width: 680px;
    max-height: 85vh;
    overflow-y: auto;
    box-shadow:
        0 25px 60px rgba(0,0,0,0.6),
        0 0 0 1px var(--gold-dim);
    animation: orderIn 0.35s ease;
}

@keyframes orderIn {
    from { opacity: 0; transform: scale(0.96) translateY(15px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}

.order-header {
    padding: 1.2rem 1.5rem;
    border-bottom: 1px solid var(--border);
    text-align: center;
    position: relative;
}

.order-header h3 {
    font-family: var(--font-serif);
    font-size: 1rem;
    letter-spacing: 4px;
    color: var(--gold);
}

.order-header-ornament {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin: 0.3rem 0;
}

.oh-line {
    width: 60px;
    height: 1px;
    background: var(--gold-dim);
}

.oh-diamond {
    color: var(--gold);
    font-size: 0.5rem;
}

.order-close {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 30px;
    height: 30px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    transition: var(--transition);
    display: flex;
    align-items: center;
    justify-content: center;
}

.order-close:hover {
    border-color: var(--verdict-red-dim);
    color: var(--verdict-red-bright);
    background: var(--verdict-red-dim);
}

.order-body {
    padding: 1.5rem;
}

.order-section {
    margin-bottom: 1.5rem;
}

.order-section:last-child { margin-bottom: 0; }

.order-section-title {
    font-family: var(--font-serif);
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: var(--gold);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.order-section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--gold-dim);
}

.order-field {
    background: var(--parchment-mid);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.9rem;
    margin-bottom: 0.5rem;
}

.order-field .of-label {
    font-family: var(--font-mono);
    font-size: 0.55rem;
    letter-spacing: 2px;
    color: var(--text-dim);
    margin-bottom: 0.3rem;
    text-transform: uppercase;
}

.order-field .of-value {
    font-family: var(--font-body);
    font-size: 0.95rem;
    color: var(--text-bright);
    line-height: 1.5;
}

.order-verdict-banner {
    padding: 1rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-serif);
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
    letter-spacing: 3px;
}

.order-verdict-banner.verified {
    background: var(--seal-green-dim);
    color: var(--seal-green-bright);
    border: 1px solid var(--seal-green-dim);
}

.order-verdict-banner.hallucinated {
    background: var(--verdict-red-dim);
    color: var(--verdict-red-bright);
    border: 1px solid var(--verdict-red-dim);
}

.order-verdict-banner.skipped {
    background: var(--royal-blue-dim);
    color: var(--royal-blue-bright);
    border: 1px solid var(--royal-blue-dim);
}

.order-verdict-banner.no-match {
    background: var(--amber-dim);
    color: var(--amber);
    border: 1px solid var(--amber-dim);
}

.order-footer {
    padding: 1.2rem 1.5rem;
    border-top: 1px solid var(--border);
    text-align: center;
}

.order-seal {
    margin-bottom: 0.75rem;
}

.stamp-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 70px;
    height: 70px;
    border-radius: 50%;
    border: 2px solid var(--gold);
    font-family: var(--font-mono);
    font-size: 0.5rem;
    letter-spacing: 1px;
    color: var(--gold);
    transform: rotate(-15deg);
    opacity: 0.5;
}

.order-disclaimer {
    font-size: 0.7rem;
    color: var(--text-dim);
    font-style: italic;
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ==========================================
   FOOTER / COURT REGISTRY
   ========================================== */
.court-registry {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 1.5rem;
    background: var(--glass);
    backdrop-filter: blur(16px);
    border-top: 1px solid var(--border);
    font-size: 0.6rem;
    letter-spacing: 1px;
    color: var(--text-dim);
    position: relative;
}

.court-registry::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
}

.registry-badge {
    padding: 0.2rem 0.6rem;
    background: var(--mahogany);
    border: 1px solid var(--gold-dim);
    border-radius: 100px;
    color: var(--gold);
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: 0.55rem;
}

.registry-center {
    font-family: var(--font-serif);
    font-style: italic;
    font-size: 0.75rem;
    color: var(--text-dim);
}

.registry-sep {
    margin: 0 0.3rem;
    color: var(--gold-dim);
}

.registry-right {
    font-family: var(--font-mono);
}

/* ==========================================
   TOAST NOTIFICATIONS
   ========================================== */
.toast-container {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.toast {
    padding: 0.8rem 1.2rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
    animation: toastIn 0.3s ease, toastOut 0.3s ease 3.2s forwards;
    max-width: 400px;
}

.toast.success {
    background: var(--seal-green-dim);
    border: 1px solid var(--seal-green-dim);
    color: var(--seal-green-bright);
}

.toast.error {
    background: var(--verdict-red-dim);
    border: 1px solid var(--verdict-red-dim);
    color: var(--verdict-red-bright);
}

.toast.info {
    background: var(--gold-dim);
    border: 1px solid var(--gold-dim);
    color: var(--gold-bright);
}

.toast.warning {
    background: var(--amber-dim);
    border: 1px solid var(--amber-dim);
    color: var(--amber);
}

@keyframes toastIn {
    from { opacity: 0; transform: translateX(30px); }
    to   { opacity: 1; transform: translateX(0); }
}

@keyframes toastOut {
    from { opacity: 1; }
    to   { opacity: 0; transform: translateX(30px); }
}

/* ==========================================
   RESPONSIVE
   ========================================== */
@media (max-width: 900px) {
    .courtroom {
        grid-template-columns: 1fr;
    }

    .filing-desk {
        border-right: none;
        border-bottom: 1px solid var(--border);
        max-height: none;
    }

    .judgment-chamber { max-height: none; }
    .court-panel { max-height: none; }
    .bench-status-row { display: none; }
    .bench-subtitle { display: none; }
}

@media (max-width: 600px) {
    .verdict-summary { grid-template-columns: 1fr 1fr; }
    .verdict-summary .verdict-card.total { grid-column: 1 / -1; }
    .bench-bar { padding: 0.5rem 0.75rem; }
    .court-panel { padding: 1rem; }
    .bench-session { display: none; }
    .court-registry { flex-direction: column; gap: 0.3rem; text-align: center; }
    .judgment-filters { overflow-x: auto; flex-wrap: nowrap; }
}

/* ==========================================
   V2.0 NEW FEATURES
   ========================================== */

/* ── Version Badge ── */
.version-badge {
    font-family: var(--font-mono);
    font-size: 0.5rem;
    background: var(--gold-dim);
    color: var(--gold);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    letter-spacing: 1px;
    vertical-align: middle;
    margin-left: 0.4rem;
}

/* ── Nav Tabs ── */
.bench-nav-tabs {
    display: flex;
    gap: 0.25rem;
    background: var(--parchment-mid);
    padding: 0.25rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
}

.nav-tab {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.75rem;
    border: none;
    border-radius: calc(var(--radius) - 2px);
    background: transparent;
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 1.5px;
    cursor: pointer;
    transition: var(--transition);
}
.nav-tab:hover { color: var(--text-primary); background: var(--parchment); }
.nav-tab.active {
    background: var(--mahogany);
    color: var(--gold-bright);
    border: 1px solid var(--gold-dim);
    box-shadow: 0 0 12px var(--gold-dim);
}

/* ── Confidence Bar ── */
.confidence-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.75rem;
    background: rgba(0,0,0,0.2);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}
.confidence-label {
    font-family: var(--font-mono);
    font-size: 0.55rem;
    color: var(--text-dim);
    letter-spacing: 1px;
    white-space: nowrap;
}
.confidence-track {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 6px;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.confidence-fill.high { background: linear-gradient(90deg, #2d6a4f, #40916c); box-shadow: 0 0 8px #40916c55; }
.confidence-fill.mid  { background: linear-gradient(90deg, #b5651d, var(--amber)); }
.confidence-fill.low  { background: linear-gradient(90deg, var(--verdict-red), var(--verdict-red-bright)); }
.confidence-pct {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-secondary);
    min-width: 30px;
    text-align: right;
}

/* ── Post Audit Action Buttons ── */
.post-audit-actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.action-btn {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 0.9rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: var(--parchment-mid);
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 1px;
    cursor: pointer;
    transition: var(--transition);
    flex: 1;
    justify-content: center;
}
.action-btn:hover { border-color: var(--gold-dim); color: var(--gold); background: var(--mahogany); }
.export-pdf-btn:hover { border-color: rgba(220,50,50,0.4); color: #e87777; }
.export-csv-btn:hover { border-color: rgba(76,175,130,0.4); color: #4caf8a; }
.summary-btn:hover { border-color: var(--gold-dim); color: var(--gold); }

/* ── Search Tab ── */
.search-tab, .bulk-tab, .history-tab {
    display: flex;
    flex: 1;
    overflow: hidden;
}
.search-panel, .bulk-panel, .history-panel {
    flex: 1;
    border: none;
    max-height: calc(100vh - 100px);
    overflow-y: auto;
}
.search-panel-body, .bulk-panel-body {
    padding: 0.5rem 0;
}
.search-description {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
    line-height: 1.6;
}
.search-input-group {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.search-input-wrap {
    flex: 1;
    position: relative;
}
.search-icon {
    position: absolute;
    left: 0.85rem;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-dim);
    font-size: 0.8rem;
}
.manual-search-input {
    width: 100%;
    padding: 0.75rem 0.75rem 0.75rem 2.5rem;
    background: var(--parchment-mid);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-family: var(--font-body);
    font-size: 1rem;
    transition: var(--transition);
}
.manual-search-input:focus {
    outline: none;
    border-color: var(--gold-dim);
    box-shadow: 0 0 20px var(--gold-dim);
}
.manual-search-input::placeholder { color: var(--text-dim); }
.search-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: var(--mahogany);
    border: 2px solid var(--gold);
    border-radius: var(--radius-sm);
    color: var(--gold-bright);
    font-family: var(--font-serif);
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 2px;
    cursor: pointer;
    transition: var(--transition);
    white-space: nowrap;
}
.search-btn:hover:not(:disabled) {
    background: var(--mahogany-light);
    box-shadow: 0 0 20px var(--gold-dim);
}
.search-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.search-examples {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}
.ex-label { font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-dim); }
.ex-chip {
    padding: 0.25rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: 100px;
    background: var(--parchment-mid);
    color: var(--text-secondary);
    font-size: 0.7rem;
    cursor: pointer;
    transition: var(--transition);
    font-family: var(--font-body);
}
.ex-chip:hover { border-color: var(--gold-dim); color: var(--gold); }

.search-results-area { min-height: 300px; }
.search-idle, .search-loading, .search-error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 3rem;
    color: var(--text-dim);
    text-align: center;
}
.search-result-card {
    border-radius: var(--radius);
    border: 1px solid var(--border);
    overflow: hidden;
    animation: slideUp 0.4s ease;
    background: var(--parchment-mid);
}
.search-result-card.verified { border-color: var(--seal-green-dim); }
.search-result-card.hallucinated { border-color: var(--verdict-red-dim); }
.search-result-card.skipped { border-color: var(--amber-dim); }
.src-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding: 1rem 1.25rem;
    gap: 1rem;
}
.src-citation {
    font-family: var(--font-serif);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-bright);
    line-height: 1.4;
}
.src-verdict {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    white-space: nowrap;
}
.src-details { padding: 0.75rem 1.25rem 1.25rem; display: flex; flex-direction: column; gap: 0.5rem; }
.src-field { display: flex; gap: 1rem; font-size: 0.85rem; }
.src-label { font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 1.5px; color: var(--text-dim); min-width: 60px; padding-top: 2px; }

/* ── Bulk Upload ── */
.bulk-dropzone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 2.5rem;
    text-align: center;
    cursor: pointer;
    transition: var(--transition);
    background: var(--parchment-mid);
}
.bulk-dropzone:hover, .bulk-dropzone.drag-over {
    border-color: var(--gold);
    box-shadow: 0 0 30px var(--gold-dim);
}
.bulk-dropzone i { font-size: 2.5rem; color: var(--gold); margin-bottom: 1rem; display: block; }
.bulk-dropzone h3 { font-family: var(--font-serif); letter-spacing: 3px; color: var(--gold-bright); margin-bottom: 0.5rem; }
.bulk-dropzone p { color: var(--text-dim); font-size: 0.85rem; }

.bulk-file-list { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.5rem; }
.bulk-file-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.75rem;
    background: var(--parchment-mid);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
}
.bfi-name { flex: 1; color: var(--text-bright); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bfi-size { font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-dim); }
.bfi-remove { background: transparent; border: none; color: var(--text-dim); cursor: pointer; padding: 0.25rem; border-radius: 3px; }
.bfi-remove:hover { color: #e87777; }

.bulk-progress { margin: 1rem 0; }
.bulk-progress-bar { height: 6px; background: var(--border); border-radius: 6px; overflow: hidden; margin-bottom: 0.5rem; }
.bulk-progress-fill { height: 100%; background: linear-gradient(90deg, var(--gold), var(--gold-bright)); border-radius: 6px; transition: width 0.5s ease; width: 0%; }
.bulk-progress-text { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-secondary); }

.bulk-results-area { margin-top: 1.5rem; display: flex; flex-direction: column; gap: 0.75rem; }
.bulk-summary-header {
    display: flex;
    gap: 1rem;
    padding: 1rem;
    background: var(--parchment-mid);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 0.5rem;
}
.bulk-stat { text-align: center; flex: 1; }
.bulk-stat .bs-num { font-family: var(--font-serif); font-size: 2rem; font-weight: 800; color: var(--gold); }
.bulk-stat .bs-label { font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 1.5px; color: var(--text-dim); }
.bulk-stat.upheld .bs-num { color: #4caf8a; }
.bulk-stat.fabricated .bs-num { color: #e87777; }
.bulk-doc-card {
    background: var(--parchment-mid);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
}
.bulk-doc-card.error { border-color: var(--verdict-red-dim); }
.bdc-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
.bdc-name { flex: 1; font-weight: 600; color: var(--text-bright); }
.bdc-badge {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    padding: 0.15rem 0.5rem;
    border-radius: 100px;
    background: var(--gold-dim);
    color: var(--gold);
}
.bdc-badge.error { background: var(--verdict-red-dim); color: var(--verdict-red-bright); }
.bdc-stats { display: flex; gap: 1rem; font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-secondary); }

/* ── History Tab ── */
.history-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.history-count { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); }
.history-list { display: flex; flex-direction: column; gap: 0.5rem; }
.history-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    padding: 3rem;
    color: var(--text-dim);
    text-align: center;
}
.history-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    background: var(--parchment-mid);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: var(--transition);
}
.history-item:hover { border-color: var(--gold-dim); background: var(--mahogany); }
.hi-left { display: flex; align-items: center; gap: 0.75rem; flex: 1; min-width: 0; }
.hi-icon { font-size: 1.2rem; color: var(--gold); flex-shrink: 0; }
.hi-info { min-width: 0; }
.hi-filename { font-weight: 600; color: var(--text-bright); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hi-date { font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-dim); margin-top: 0.2rem; }
.hi-stats { display: flex; gap: 0.75rem; font-family: var(--font-mono); font-size: 0.65rem; }
.hi-stat { color: var(--text-secondary); }
.hi-delete {
    background: transparent;
    border: none;
    color: var(--text-dim);
    cursor: pointer;
    padding: 0.4rem;
    border-radius: 5px;
    flex-shrink: 0;
    transition: var(--transition);
}
.hi-delete:hover { color: #e87777; background: rgba(220,50,50,0.1); }

/* ── Summary Modal Content ── */
.summary-loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    padding: 3rem;
    color: var(--text-secondary);
}
.summary-risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 1rem;
}
.summary-stats-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.sum-stat {
    flex: 1;
    text-align: center;
    padding: 0.75rem;
    background: var(--parchment-mid);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
}
.sum-stat span { display: block; font-family: var(--font-serif); font-size: 1.8rem; font-weight: 800; color: var(--text-secondary); }
.sum-stat > span:last-child { font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-dim); letter-spacing: 1.5px; }
.sum-stat.up span:first-child { color: #4caf8a; }
.sum-stat.fab span:first-child { color: #e87777; }
.sum-stat.sk span:first-child { color: var(--amber); }
.summary-text {
    font-family: var(--font-body);
    font-size: 1rem;
    line-height: 1.8;
    color: var(--text-primary);
    border-left: 3px solid var(--gold-dim);
    padding-left: 1rem;
}
.summary-text p { margin-bottom: 0.75rem; }

/* ── Chatbot Widget ── */
.chat-bubble {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--mahogany), #3a2a1c);
    border: 2px solid var(--gold);
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 999;
    box-shadow: 0 4px 20px var(--gold-dim), 0 0 40px rgba(201,168,76,0.1);
    transition: var(--transition);
}
.chat-bubble:hover { transform: translateY(-3px) scale(1.05); box-shadow: 0 8px 30px var(--gold-dim); }
.chat-bubble-icon { font-size: 1.5rem; line-height: 1; }
.chat-bubble-label { font-family: var(--font-mono); font-size: 0.45rem; color: var(--gold); letter-spacing: 1px; }
.chat-notification {
    position: absolute;
    top: -4px;
    right: -4px;
    background: var(--verdict-red-bright);
    color: #fff;
    font-size: 0.7rem;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
}

.chat-pane {
    position: fixed;
    bottom: 5.5rem;
    right: 1.5rem;
    width: 380px;
    max-height: 560px;
    background: var(--parchment-light);
    border: 1px solid var(--gold-dim);
    border-radius: 16px;
    z-index: 998;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 40px var(--gold-dim);
    animation: chatSlideUp 0.3s cubic-bezier(0.4,0,0.2,1);
    overflow: hidden;
}

@keyframes chatSlideUp {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1rem;
    background: var(--mahogany);
    border-bottom: 1px solid var(--gold-dim);
}
.chat-header-left { display: flex; align-items: center; gap: 0.75rem; }
.chat-avatar { font-size: 1.5rem; }
.chat-title { font-family: var(--font-serif); font-weight: 700; font-size: 0.95rem; color: var(--gold-bright); }
.chat-subtitle { font-family: var(--font-mono); font-size: 0.55rem; color: var(--text-dim); letter-spacing: 1px; }
.chat-header-actions { display: flex; gap: 0.25rem; }
.chat-action-btn {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-dim);
    cursor: pointer;
    transition: var(--transition);
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    justify-content: center;
}
.chat-action-btn:hover { background: var(--border); color: var(--text-primary); border-color: var(--border); }

.chat-context-bar {
    padding: 0.4rem 0.85rem;
    background: rgba(201,168,76,0.08);
    border-bottom: 1px solid var(--gold-dim);
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--gold);
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
}
.chat-msg { display: flex; gap: 0.5rem; align-items: flex-end; }
.chat-msg.user { flex-direction: row-reverse; }
.msg-avatar { font-size: 1.2rem; flex-shrink: 0; }
.msg-bubble {
    max-width: 85%;
    padding: 0.65rem 0.9rem;
    border-radius: 12px;
    font-size: 0.88rem;
    line-height: 1.6;
}
.chat-msg.assistant .msg-bubble {
    background: var(--parchment-mid);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
}
.chat-msg.user .msg-bubble {
    background: var(--mahogany);
    border: 1px solid var(--gold-dim);
    color: var(--gold-bright);
    border-bottom-right-radius: 4px;
}

.typing-dots { display: flex; gap: 4px; padding: 4px 0; }
.typing-dots span {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--text-dim);
    animation: typingDot 1.4s ease infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingDot { 0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }

.chat-suggestions {
    padding: 0.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    border-top: 1px solid var(--border);
}
.chat-chip {
    padding: 0.25rem 0.65rem;
    border: 1px solid var(--border);
    border-radius: 100px;
    background: var(--parchment-mid);
    color: var(--text-secondary);
    font-size: 0.72rem;
    cursor: pointer;
    transition: var(--transition);
    font-family: var(--font-body);
}
.chat-chip:hover { border-color: var(--gold-dim); color: var(--gold); }

.chat-input-area {
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
    padding: 0.75rem;
    border-top: 1px solid var(--border);
    background: var(--parchment-mid);
}
.chat-input {
    flex: 1;
    background: var(--parchment);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-family: var(--font-body);
    font-size: 0.9rem;
    padding: 0.6rem 0.8rem;
    resize: none;
    min-height: 38px;
    max-height: 120px;
    transition: var(--transition);
    line-height: 1.4;
}
.chat-input:focus { outline: none; border-color: var(--gold-dim); box-shadow: 0 0 12px var(--gold-dim); }
.chat-input::placeholder { color: var(--text-dim); }
.chat-send-btn {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    border: 2px solid var(--gold);
    background: var(--mahogany);
    color: var(--gold-bright);
    cursor: pointer;
    transition: var(--transition);
    flex-shrink: 0;
    font-size: 0.85rem;
}
.chat-send-btn:hover:not(:disabled) {
    background: var(--mahogany-light);
    box-shadow: 0 0 15px var(--gold-dim);
}
.chat-send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Responsive for new tabs ── */
@media (max-width: 768px) {
    .chat-pane { width: calc(100vw - 2rem); right: 1rem; }
    .bench-nav-tabs { gap: 0.15rem; }
    .nav-tab span { display: none; }
}

```

---

### File: `frontend/app.js`

```js
// ==========================================
// ⚖️ LEGAL CITATION AUDITOR v2.0 — ENGINE
// Features: Audit, Search, Bulk, History, Export, Chatbot, Summary
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
const sessionId        = document.getElementById('session-id');
const toastContainer   = document.getElementById('toast-container');

// Stats
const vcTotal      = document.getElementById('vc-total');
const vcUpheld     = document.getElementById('vc-upheld');
const vcOverruled  = document.getElementById('vc-overruled');
const vcSkipped    = document.getElementById('vc-skipped');
const vcUnheard    = document.getElementById('vc-unheard');

// Filter counts
const jfAll        = document.getElementById('jf-all');
const jfUpheld     = document.getElementById('jf-upheld');
const jfFabricated = document.getElementById('jf-fabricated');
const jfSkipped    = document.getElementById('jf-skipped');
const jfUnverified = document.getElementById('jf-unverified');

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
    setTimeout(() => {
        oathScreen.classList.add('dismissed');
        appEl.classList.remove('hidden');
        setTimeout(() => { oathScreen.style.display = 'none'; }, 1000);
    }, 4000);
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

sessionId.textContent = `SCI-${Date.now().toString(36).toUpperCase().slice(-6)}`;

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

    ['ds-1','ds-2','ds-3','ds-4','ds-5'].forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
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
    const steps = ['ds-1', 'ds-2', 'ds-3', 'ds-4', 'ds-5'];
    const texts = [
        ['READING DOCUMENT', 'Extracting text from filed PDF...'],
        ['IDENTIFYING AUTHORITIES', 'AI is finding all cited cases...'],
        ['CLASSIFYING COURTS', 'Separating High Court and Supreme Court citations...'],
        ['SEARCHING COURT RECORDS', 'Cross-referencing SC cases against the archive...'],
        ['PRONOUNCING JUDGMENT', 'Running hallucination detection...']
    ];
    steps.forEach((id, i) => {
        setTimeout(() => {
            if (i > 0) {
                document.getElementById(steps[i - 1]).classList.remove('active');
                document.getElementById(steps[i - 1]).classList.add('done');
            }
            document.getElementById(id).classList.add('active');
            document.getElementById('delib-title').textContent = texts[i][0];
            document.getElementById('delib-sub').textContent = texts[i][1];
        }, i * 1800);
    });
}

function finishDeliberation() {
    ['ds-1','ds-2','ds-3','ds-4','ds-5'].forEach(id => {
        document.getElementById(id).classList.remove('active');
        document.getElementById(id).classList.add('done');
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
    judgmentRoll.innerHTML = '';

    results.forEach((item, i) => {
        const status = classifyStatus(item);
        if (status === 'verified')      upheld++;
        else if (status === 'hallucinated') fabricated++;
        else if (status === 'skipped')  skipped++;
        else                            unverified++;
        const card = buildJudgmentCard(item, i, status);
        judgmentRoll.appendChild(card);
    });

    animateNum(vcTotal, total);
    animateNum(vcUpheld, upheld);
    animateNum(vcOverruled, fabricated);
    animateNum(vcSkipped, skipped);
    animateNum(vcUnheard, unverified);

    jfAll.textContent        = total;
    jfUpheld.textContent     = upheld;
    jfFabricated.textContent = fabricated;
    jfSkipped.textContent    = skipped;
    jfUnverified.textContent = unverified;

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
    if (raw.includes('skipped') || raw.includes('⚠️')) return 'skipped';
    if (courtType === 'high court' && !raw.includes('verified') && !raw.includes('hallucination')) return 'skipped';
    if (raw.includes('verified') || raw.includes('🟢')) return 'verified';
    if (raw.includes('hallucination') || raw.includes('🔴')) return 'hallucinated';
    return 'no-match';
}

function buildJudgmentCard(item, index, status) {
    const card = document.createElement('div');
    card.className = `j-card ${status}`;
    card.style.animationDelay = `${index * 0.08}s`;
    card.dataset.status = status;

    const v = item.verification || {};
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

    const confidenceBar = confidence !== null ? `
        <div class="confidence-bar-wrap">
            <span class="confidence-label">AI Confidence</span>
            <div class="confidence-track">
                <div class="confidence-fill ${confidence >= 80 ? 'high' : confidence >= 50 ? 'mid' : 'low'}" 
                     style="width:${confidence}%"></div>
            </div>
            <span class="confidence-pct">${confidence}%</span>
        </div>` : '';

    card.innerHTML = `
        <div class="j-card-top">
            <div class="j-case-name">${esc(item.target_citation)}</div>
            <span class="j-verdict-badge">${verdictLabels[status]}</span>
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
            card.style.display = (filter === 'all' || card.dataset.status === filter) ? '' : 'none';
        });
    });
});

// ==========================================
// 8. ORDER MODAL
// ==========================================
function openOrder(item, status) {
    const v = item.verification || {};
    const courtType = item.court_type || 'Unknown';
    const confidence = typeof v.confidence === 'number' ? v.confidence : null;

    const verdictText = {
        'verified': '✅ CITATION UPHELD — Exists in Supreme Court Records',
        'hallucinated': '❌ CITATION FABRICATED — Not Found in Any Record',
        'skipped': '⚠️ HIGH COURT CITATION — Bypassed Supreme Court Verification',
        'no-match': '⚠️ CITATION UNVERIFIED — No Matching Candidates Found'
    };

    let sectionIdx = 0;
    const nextSection = () => ['I','II','III','IV','V','VI'][sectionIdx++];
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
                <pre style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-secondary);white-space:pre-wrap;word-break:break-all;line-height:1.6;margin:0;">${esc(JSON.stringify({ court_type: courtType, verification: v }, null, 2))}</pre>
            </div>
        </div>`;

    orderBody.innerHTML = html;
    orderOverlay.classList.remove('hidden');
}

orderClose.addEventListener('click', () => orderOverlay.classList.add('hidden'));
orderOverlay.addEventListener('click', (e) => { if (e.target === orderOverlay) orderOverlay.classList.add('hidden'); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { orderOverlay.classList.add('hidden'); document.getElementById('summary-overlay').classList.add('hidden'); } });

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
    bulkFiles = [...bulkFiles, ...pdfs].slice(0, 10); // Max 10 files
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

    container.innerHTML = `
        <div class="bulk-summary-header">
            <div class="bulk-stat"><div class="bs-num">${results.length}</div><div class="bs-label">Documents</div></div>
            <div class="bulk-stat"><div class="bs-num">${totalCitations}</div><div class="bs-label">Citations</div></div>
            <div class="bulk-stat upheld"><div class="bs-num">${totalVerified}</div><div class="bs-label">Verified</div></div>
            <div class="bulk-stat fabricated"><div class="bs-num">${totalFabricatedC}</div><div class="bs-label">Fabricated</div></div>
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
            </div>` : `<p style="color:#e87777;font-size:0.8rem;">${esc(r.error)}</p>`}
        </div>`).join('')}`;
}

// ==========================================
// 11. EXPORT FUNCTIONS
// ==========================================
function exportCSV() {
    if (!auditData.length) { showToast('No audit data to export.', 'warning'); return; }
    const rows = [['#', 'Citation', 'Court Type', 'Status', 'Confidence', 'Matched Name', 'Reason/Message']];
    auditData.forEach((item, i) => {
        const v = item.verification || {};
        rows.push([
            i + 1,
            `"${(item.target_citation || '').replace(/"/g, '""')}"`,
            `"${(item.court_type || '').replace(/"/g, '""')}"`,
            `"${(v.status || '').replace(/"/g, '""')}"`,
            v.confidence ?? '',
            `"${(v.matched_name || '').replace(/"/g, '""')}"`,
            `"${(v.reason || v.message || '').replace(/"/g, '""')}"`
        ]);
    });
    const csv = rows.map(r => r.join(',')).join('\n');
    downloadFile('audit-report.csv', csv, 'text/csv');
    showToast('CSV report downloaded.', 'success');
}

function exportPDF() {
    if (!auditData.length) { showToast('No audit data to export.', 'warning'); return; }
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const pageWidth = doc.internal.pageSize.getWidth();
    let y = 20;

    // Header
    doc.setFillColor(15, 12, 30);
    doc.rect(0, 0, pageWidth, 40, 'F');
    doc.setTextColor(212, 175, 55);
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text('LEGAL CITATION AUDIT REPORT', pageWidth / 2, 18, { align: 'center' });
    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(180, 180, 200);
    doc.text('Supreme Court of India • AI Citation Integrity Verification', pageWidth / 2, 26, { align: 'center' });
    doc.text(`Generated: ${new Date().toLocaleString('en-IN')} • Session: ${sessionId.textContent}`, pageWidth / 2, 33, { align: 'center' });
    y = 55;

    // Stats box
    const total = auditData.length;
    const verified = auditData.filter(r => classifyStatus(r) === 'verified').length;
    const fabricated = auditData.filter(r => classifyStatus(r) === 'hallucinated').length;
    const skipped = auditData.filter(r => classifyStatus(r) === 'skipped').length;
    const unverified = total - verified - fabricated - skipped;

    doc.setFillColor(30, 25, 60);
    doc.roundedRect(10, y, pageWidth - 20, 28, 3, 3, 'F');
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');

    const statItems = [
        { label: 'TOTAL', val: total, color: [212, 175, 55] },
        { label: 'UPHELD', val: verified, color: [76, 175, 130] },
        { label: 'FABRICATED', val: fabricated, color: [232, 119, 119] },
        { label: 'HC', val: skipped, color: [232, 184, 119] },
        { label: 'UNVERIFIED', val: unverified, color: [170, 170, 170] }
    ];
    statItems.forEach((s, i) => {
        const x = 15 + i * 38;
        doc.setTextColor(...s.color);
        doc.setFontSize(14);
        doc.text(String(s.val), x, y + 14);
        doc.setFontSize(7);
        doc.setTextColor(150, 150, 170);
        doc.text(s.label, x, y + 22);
    });
    y += 38;

    // Citation rows
    auditData.forEach((item, i) => {
        if (y > 260) { doc.addPage(); y = 20; }
        const status = classifyStatus(item);
        const v = item.verification || {};
        const bgColors = {
            'verified': [20, 50, 35], 'hallucinated': [60, 20, 20],
            'skipped': [50, 40, 10], 'no-match': [30, 30, 50]
        };
        doc.setFillColor(...(bgColors[status] || [30, 30, 60]));
        doc.roundedRect(10, y, pageWidth - 20, 22, 2, 2, 'F');

        doc.setFontSize(8);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(212, 175, 55);
        doc.text(`#${String(i+1).padStart(3,'0')}`, 14, y + 8);

        const citation = (item.target_citation || '').substring(0, 65);
        doc.setTextColor(230, 230, 240);
        doc.text(citation + (item.target_citation.length > 65 ? '...' : ''), 28, y + 8);

        const statusColors2 = {
            'verified': [76, 200, 130], 'hallucinated': [232, 100, 100],
            'skipped': [232, 184, 119], 'no-match': [150, 150, 170]
        };
        const statusLabels2 = { 'verified': 'UPHELD', 'hallucinated': 'FABRICATED', 'skipped': 'HC', 'no-match': 'UNVERIFIED' };
        doc.setTextColor(...(statusColors2[status] || [150, 150, 170]));
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(7);
        doc.text(statusLabels2[status] || 'UNKNOWN', pageWidth - 12, y + 8, { align: 'right' });

        if (v.matched_name) {
            doc.setFont('helvetica', 'italic');
            doc.setTextColor(170, 170, 200);
            doc.setFontSize(7);
            doc.text(`Match: ${v.matched_name.substring(0, 80)}`, 28, y + 15);
        }
        if (typeof v.confidence === 'number') {
            doc.setFont('helvetica', 'normal');
            doc.setTextColor(150, 150, 170);
            doc.text(`Confidence: ${v.confidence}%`, pageWidth - 12, y + 15, { align: 'right' });
        }
        y += 26;
    });

    // Footer
    doc.setFontSize(7);
    doc.setTextColor(100, 100, 130);
    doc.text('Powered by Groq × LLaMA 3.3 70B • This report is for informational purposes only and does not constitute legal advice.', pageWidth / 2, 290, { align: 'center' });

    doc.save(`legal-audit-${Date.now()}.pdf`);
    showToast('PDF report downloaded.', 'success');
}

function exportSummaryPDF() {
    const bodyEl = document.getElementById('summary-body');
    const text = bodyEl.innerText;
    if (!text || text.includes('Generating')) { showToast('Summary not ready yet.', 'warning'); return; }
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Legal Audit Summary Report', 20, 20);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(80, 80, 80);
    const lines = doc.splitTextToSize(text, 170);
    doc.text(lines, 20, 35);
    doc.save(`audit-summary-${Date.now()}.pdf`);
}

function downloadFile(name, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
}

// ==========================================
// 12. AI SUMMARY
// ==========================================
async function generateSummary() {
    if (!lastAuditResponse) { showToast('Run an audit first.', 'warning'); return; }
    const overlay = document.getElementById('summary-overlay');
    const body = document.getElementById('summary-body');
    overlay.classList.remove('hidden');
    body.innerHTML = `<div class="summary-loading"><i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Generating professional summary...</p></div>`;

    try {
        const resp = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: lastAuditResponse.results || [],
                total: lastAuditResponse.total_citations_found || 0,
                sc_count: lastAuditResponse.supreme_court_count || 0,
                hc_count: lastAuditResponse.high_court_count || 0
            })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        const riskColors = { 'Low': '#4caf8a', 'Medium': '#e8b877', 'High': '#e87777' };
        const riskColor = riskColors[data.risk_level] || '#aaa';

        body.innerHTML = `
            <div class="summary-risk-badge" style="background:${riskColor}22;border:1px solid ${riskColor};color:${riskColor};">
                <i class="fas fa-shield-alt"></i> RISK LEVEL: ${esc(data.risk_level)}
            </div>
            <div class="summary-stats-row">
                <div class="sum-stat up"><span>${data.stats.verified}</span>Verified</div>
                <div class="sum-stat fab"><span>${data.stats.fabricated}</span>Fabricated</div>
                <div class="sum-stat sk"><span>${data.stats.skipped}</span>HC</div>
                <div class="sum-stat un"><span>${data.stats.unverified}</span>Unverified</div>
            </div>
            <div class="summary-text">${esc(data.summary).replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</div>`;
    } catch (err) {
        body.innerHTML = `<p style="color:#e87777;">Failed to generate summary: ${esc(err.message)}</p>`;
    }
}

// ==========================================
// 13. AUDIT HISTORY
// ==========================================
function saveToHistory(data, filename) {
    const history = JSON.parse(localStorage.getItem('lca_history') || '[]');
    const entry = {
        id: Date.now(),
        filename,
        timestamp: new Date().toISOString(),
        total: data.total_citations_found || 0,
        sc: data.supreme_court_count || 0,
        hc: data.high_court_count || 0,
        verified:    (data.results || []).filter(r => classifyStatus(r) === 'verified').length,
        fabricated:  (data.results || []).filter(r => classifyStatus(r) === 'hallucinated').length,
        skipped:     (data.results || []).filter(r => classifyStatus(r) === 'skipped').length,
        results:     data.results || []
    };
    history.unshift(entry);
    if (history.length > 50) history.pop();
    localStorage.setItem('lca_history', JSON.stringify(history));
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem('lca_history') || '[]');
    const list = document.getElementById('history-list');
    const countEl = document.getElementById('history-count');
    countEl.textContent = `${history.length} record${history.length !== 1 ? 's' : ''}`;

    if (!history.length) {
        list.innerHTML = `<div class="history-empty"><i class="fas fa-history" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i><p>No audit history yet.</p></div>`;
        return;
    }

    list.innerHTML = history.map(entry => `
        <div class="history-item" onclick="restoreFromHistory('${entry.id}')">
            <div class="hi-left">
                <div class="hi-icon"><i class="fas fa-file-contract"></i></div>
                <div class="hi-info">
                    <div class="hi-filename">${esc(entry.filename)}</div>
                    <div class="hi-date">${new Date(entry.timestamp).toLocaleString('en-IN')}</div>
                </div>
            </div>
            <div class="hi-stats">
                <span class="hi-stat up" title="Verified">✅ ${entry.verified}</span>
                <span class="hi-stat fab" title="Fabricated">❌ ${entry.fabricated}</span>
                <span class="hi-stat hc" title="High Court">🏛️ ${entry.hc}</span>
                <span class="hi-stat total" title="Total">${entry.total} total</span>
            </div>
            <button class="hi-delete" onclick="event.stopPropagation();deleteHistoryItem(${entry.id})" title="Delete">
                <i class="fas fa-trash"></i>
            </button>
        </div>`).join('');
}

function restoreFromHistory(id) {
    const history = JSON.parse(localStorage.getItem('lca_history') || '[]');
    const entry = history.find(h => h.id == id);
    if (!entry) return;
    auditData = entry.results;
    lastAuditResponse = {
        results: entry.results,
        total_citations_found: entry.total,
        supreme_court_count: entry.sc,
        high_court_count: entry.hc
    };
    switchTab('audit');
    renderJudgments(lastAuditResponse);
    showToast(`Loaded audit: ${entry.filename}`, 'info');
}

function deleteHistoryItem(id) {
    let history = JSON.parse(localStorage.getItem('lca_history') || '[]');
    history = history.filter(h => h.id != id);
    localStorage.setItem('lca_history', JSON.stringify(history));
    renderHistory();
}

function clearHistory() {
    if (!confirm('Clear all audit history?')) return;
    localStorage.removeItem('lca_history');
    renderHistory();
    showToast('History cleared.', 'info');
}

// ==========================================
// 14. CHATBOT
// ==========================================
let chatHistory = [];
let auditContextEnabled = false;

function toggleChat() {
    const pane = document.getElementById('chat-pane');
    pane.classList.toggle('hidden');
    if (!pane.classList.contains('hidden')) {
        document.getElementById('chat-notification').style.display = 'none';
        document.getElementById('chat-input').focus();
    }
}

function toggleAuditContext() {
    auditContextEnabled = !auditContextEnabled;
    const btn = document.getElementById('ctx-btn');
    const bar = document.getElementById('chat-context-bar');
    btn.style.color = auditContextEnabled ? 'var(--gold)' : '';
    bar.classList.toggle('hidden', !auditContextEnabled);
    if (auditContextEnabled && !lastAuditResponse) {
        showToast('Run an audit first to use audit context.', 'warning');
        auditContextEnabled = false;
        btn.style.color = '';
        bar.classList.add('hidden');
    }
}

function clearChat() {
    chatHistory = [];
    document.getElementById('chat-messages').innerHTML = `
        <div class="chat-msg assistant">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble"><p>Chat cleared. How can I assist you?</p></div>
        </div>`;
}

function handleChatKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
}

function autoResizeChat(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function sendSuggestion(text) {
    document.getElementById('chat-input').value = text;
    sendChatMessage();
    document.getElementById('chat-suggestions').style.display = 'none';
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const messages = document.getElementById('chat-messages');
    const text = input.value.trim();
    if (!text) return;

    // Add user message
    chatHistory.push({ role: 'user', content: text });
    appendChatMessage('user', text, messages);
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    // Typing indicator
    const typingId = 'typing-' + Date.now();
    messages.insertAdjacentHTML('beforeend', `
        <div class="chat-msg assistant" id="${typingId}">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>
        </div>`);
    messages.scrollTop = messages.scrollHeight;

    let auditContext = null;
    if (auditContextEnabled && lastAuditResponse) {
        const v = lastAuditResponse.results?.filter(r => classifyStatus(r) === 'verified').length || 0;
        const f = lastAuditResponse.results?.filter(r => classifyStatus(r) === 'hallucinated').length || 0;
        auditContext = `Last audit found ${lastAuditResponse.total_citations_found} citations: ${v} verified, ${f} fabricated.`;
    }

    try {
        const resp = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                history: chatHistory.slice(-10),
                audit_context: auditContext
            })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        document.getElementById(typingId)?.remove();
        const reply = data.reply || 'I apologize, I could not process that request.';
        chatHistory.push({ role: 'assistant', content: reply });
        appendChatMessage('assistant', reply, messages);
    } catch (err) {
        document.getElementById(typingId)?.remove();
        appendChatMessage('assistant', `⚠️ Error: ${err.message}`, messages);
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

function appendChatMessage(role, text, container) {
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    // Format text — convert **bold** and line breaks
    const formatted = esc(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    div.innerHTML = role === 'assistant'
        ? `<div class="msg-avatar">⚖️</div><div class="msg-bubble">${formatted}</div>`
        : `<div class="msg-bubble">${formatted}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// ==========================================
// 15. UTILITIES
// ==========================================
function esc(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

function truncate(str, max) {
    if (!str) return '';
    return str.length > max ? str.substring(0, max) + '…' : str;
}

function animateNum(el, target) {
    const duration = 900;
    const startTime = performance.now();
    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(target * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function showToast(message, type = 'info') {
    const icons = { success: 'check-circle', error: 'exclamation-triangle', info: 'info-circle', warning: 'exclamation-circle' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<i class="fas fa-${icons[type] || 'info-circle'}"></i><span>${esc(message)}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3800);
}

// ==========================================
// 16. INIT
// ==========================================
document.addEventListener('DOMContentLoaded', () => { runOathSequence(); });

// API health check
fetch(`${API_BASE}/`)
    .then(r => r.json())
    .then(d => {
        const apiInd = document.getElementById('ind-api');
        apiInd.classList.add('online');
        apiInd.querySelector('span').textContent = 'API';
        // Check DB stats
        return fetch(`${API_BASE}/db-stats`);
    })
    .then(r => r.json())
    .then(d => {
        const dbInd = document.getElementById('ind-db');
        if (d.loaded) {
            dbInd.classList.add('online');
            document.getElementById('registry-records').textContent = `${d.record_count.toLocaleString()} cases in archive`;
        } else {
            dbInd.classList.remove('online');
            document.getElementById('registry-records').textContent = 'No database loaded';
        }
    })
    .catch(() => {
        const apiInd = document.getElementById('ind-api');
        apiInd.classList.remove('online');
        apiInd.querySelector('span').textContent = 'OFFLINE';
    });
```

---

### File: `frontend/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚖️ Legal Citation Auditor — Supreme Court of India</title>
    <meta name="description" content="AI-powered legal citation verification system for Indian Supreme Court and High Court cases">
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
            <div class="oath-subheading">Citation Integrity Verification System v2.0</div>
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
                    <span>Engaging Hallucination Detection Protocol...</span>
                    <span class="oath-check">✓</span>
                </div>
                <div class="oath-line" style="--delay: 1.8s">
                    <span class="oath-bullet">§</span>
                    <span>Activating LexAI Legal Chatbot...</span>
                    <span class="oath-check">✓</span>
                </div>
                <div class="oath-line" style="--delay: 2.2s">
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
                    <div class="mini-scales">⚖️</div>
                </div>
                <div class="bench-title-block">
                    <h1 class="bench-title">CITATION AUDITOR <span class="version-badge">v2.0</span></h1>
                    <p class="bench-subtitle">Supreme Court of India • AI Verification Chamber</p>
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
                </div>
            </div>
            <div class="bench-right">
                <div class="bench-nav-tabs">
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

        <!-- ===== TAB CONTENT ===== -->
        <main class="courtroom" id="tab-audit">

            <!-- LEFT: FILING DESK -->
            <section class="court-panel filing-desk">
                <div class="panel-title-bar">
                    <div class="title-bar-ornament left">━━◆</div>
                    <h2>FILING DESK</h2>
                    <div class="title-bar-ornament right">◆━━</div>
                </div>

                <!-- Seal Drop Zone -->
                <div class="seal-dropzone" id="seal-dropzone">
                    <div class="seal-visual">
                        <div class="wax-seal">
                            <div class="seal-ring r1"></div>
                            <div class="seal-ring r2"></div>
                            <div class="seal-ring r3"></div>
                            <div class="seal-center">
                                <i class="fas fa-file-pdf"></i>
                            </div>
                        </div>
                        <div class="seal-rays">
                            <div class="ray" style="--angle: 0deg"></div>
                            <div class="ray" style="--angle: 45deg"></div>
                            <div class="ray" style="--angle: 90deg"></div>
                            <div class="ray" style="--angle: 135deg"></div>
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

                <!-- Filed Document Card -->
                <div class="filed-doc hidden" id="filed-doc">
                    <div class="filed-header">
                        <div class="filed-icon">
                            <i class="fas fa-file-contract"></i>
                        </div>
                        <div class="filed-info">
                            <span class="filed-name" id="filed-name">—</span>
                            <span class="filed-size" id="filed-size">—</span>
                        </div>
                        <button class="filed-remove" id="filed-remove" title="Remove">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="filed-stamp">
                        <span>DOCUMENT FILED</span>
                        <i class="fas fa-stamp"></i>
                    </div>
                </div>

                <!-- THE GAVEL BUTTON -->
                <button class="gavel-btn" id="gavel-btn" disabled>
                    <div class="gavel-bg"></div>
                    <div class="gavel-content">
                        <span class="gavel-icon">🔨</span>
                        <span class="gavel-text">ORDER! COMMENCE AUDIT</span>
                    </div>
                    <div class="gavel-shine"></div>
                </button>

                <!-- Verdict Summary Cards -->
                <div class="verdict-summary" id="verdict-summary">
                    <div class="verdict-card total">
                        <div class="vc-icon"><i class="fas fa-scroll"></i></div>
                        <div class="vc-number" id="vc-total">—</div>
                        <div class="vc-label">CITATIONS</div>
                    </div>
                    <div class="verdict-card upheld">
                        <div class="vc-icon"><i class="fas fa-gavel"></i></div>
                        <div class="vc-number" id="vc-upheld">—</div>
                        <div class="vc-label">UPHELD</div>
                    </div>
                    <div class="verdict-card overruled">
                        <div class="vc-icon"><i class="fas fa-ban"></i></div>
                        <div class="vc-number" id="vc-overruled">—</div>
                        <div class="vc-label">FABRICATED</div>
                    </div>
                    <div class="verdict-card skipped">
                        <div class="vc-icon"><i class="fas fa-university"></i></div>
                        <div class="vc-number" id="vc-skipped">—</div>
                        <div class="vc-label">HIGH COURT</div>
                    </div>
                    <div class="verdict-card unheard">
                        <div class="vc-icon"><i class="fas fa-question-circle"></i></div>
                        <div class="vc-number" id="vc-unheard">—</div>
                        <div class="vc-label">UNVERIFIED</div>
                    </div>
                </div>

                <!-- Court Breakdown -->
                <div class="court-breakdown hidden" id="court-breakdown">
                    <div class="cb-title">
                        <i class="fas fa-balance-scale-left"></i>
                        <span>COURT CLASSIFICATION</span>
                    </div>
                    <div class="cb-row">
                        <div class="cb-bar-label">Supreme Court</div>
                        <div class="cb-bar-track">
                            <div class="cb-bar-fill sc-fill" id="cb-sc-fill"></div>
                        </div>
                        <div class="cb-bar-count" id="cb-sc-count">0</div>
                    </div>
                    <div class="cb-row">
                        <div class="cb-bar-label">High Court</div>
                        <div class="cb-bar-track">
                            <div class="cb-bar-fill hc-fill" id="cb-hc-fill"></div>
                        </div>
                        <div class="cb-bar-count" id="cb-hc-count">0</div>
                    </div>
                </div>

                <!-- Action Buttons after audit -->
                <div class="post-audit-actions hidden" id="post-audit-actions">
                    <button class="action-btn export-pdf-btn" id="export-pdf-btn" onclick="exportPDF()">
                        <i class="fas fa-file-pdf"></i> Export PDF Report
                    </button>
                    <button class="action-btn export-csv-btn" id="export-csv-btn" onclick="exportCSV()">
                        <i class="fas fa-file-csv"></i> Export CSV
                    </button>
                    <button class="action-btn summary-btn" id="summary-btn" onclick="generateSummary()">
                        <i class="fas fa-file-alt"></i> AI Summary
                    </button>
                </div>
            </section>

            <!-- RIGHT: JUDGMENT CHAMBER -->
            <section class="court-panel judgment-chamber">
                <div class="panel-title-bar">
                    <div class="title-bar-ornament left">━━◆</div>
                    <h2>JUDGMENT CHAMBER</h2>
                    <div class="title-bar-ornament right">◆━━</div>
                </div>

                <!-- IDLE STATE: Scales -->
                <div class="chamber-idle" id="chamber-idle">
                    <div class="idle-scales-container">
                        <div class="scales-beam">
                            <div class="beam-line"></div>
                            <div class="scales-pivot">⚖</div>
                            <div class="scale-pan left-pan">
                                <div class="pan-chains">
                                    <div class="chain"></div>
                                    <div class="chain"></div>
                                </div>
                                <div class="pan-dish"></div>
                                <span class="pan-label">TRUTH</span>
                            </div>
                            <div class="scale-pan right-pan">
                                <div class="pan-chains">
                                    <div class="chain"></div>
                                    <div class="chain"></div>
                                </div>
                                <div class="pan-dish"></div>
                                <span class="pan-label">FICTION</span>
                            </div>
                        </div>
                    </div>
                    <h3>THE COURT IS IN SESSION</h3>
                    <p>File a document to begin judicial review of cited authorities</p>
                </div>

                <!-- DELIBERATION STATE -->
                <div class="chamber-deliberation hidden" id="chamber-deliberation">
                    <div class="deliberation-visual">
                        <div class="quill-animation">
                            <div class="quill-body">
                                <i class="fas fa-feather-alt"></i>
                            </div>
                            <div class="ink-drops">
                                <span class="ink-drop" style="--d:0"></span>
                                <span class="ink-drop" style="--d:1"></span>
                                <span class="ink-drop" style="--d:2"></span>
                            </div>
                        </div>
                    </div>
                    <h3 class="delib-title" id="delib-title">COURT IN DELIBERATION</h3>
                    <p class="delib-sub" id="delib-sub">The bench is reviewing your document...</p>
                    <div class="delib-steps">
                        <div class="d-step active" id="ds-1">
                            <div class="d-step-marker">I</div>
                            <span>Reading the Document</span>
                        </div>
                        <div class="d-step" id="ds-2">
                            <div class="d-step-marker">II</div>
                            <span>Identifying &amp; Classifying Authorities</span>
                        </div>
                        <div class="d-step" id="ds-3">
                            <div class="d-step-marker">III</div>
                            <span>Filtering High Court Citations</span>
                        </div>
                        <div class="d-step" id="ds-4">
                            <div class="d-step-marker">IV</div>
                            <span>Verifying SC Cases Against Registry</span>
                        </div>
                        <div class="d-step" id="ds-5">
                            <div class="d-step-marker">V</div>
                            <span>Pronouncing Judgment</span>
                        </div>
                    </div>
                </div>

                <!-- RESULTS STATE -->
                <div class="chamber-results hidden" id="chamber-results">
                    <!-- Filter Bar -->
                    <div class="judgment-filters">
                        <button class="jf-tab active" data-filter="all">
                            ALL MATTERS <span class="jf-count" id="jf-all">0</span>
                        </button>
                        <button class="jf-tab" data-filter="verified">
                            UPHELD <span class="jf-count" id="jf-upheld">0</span>
                        </button>
                        <button class="jf-tab" data-filter="hallucinated">
                            FABRICATED <span class="jf-count" id="jf-fabricated">0</span>
                        </button>
                        <button class="jf-tab" data-filter="skipped">
                            HIGH COURT <span class="jf-count" id="jf-skipped">0</span>
                        </button>
                        <button class="jf-tab" data-filter="no-match">
                            UNVERIFIED <span class="jf-count" id="jf-unverified">0</span>
                        </button>
                    </div>

                    <!-- Judgment Roll -->
                    <div class="judgment-roll" id="judgment-roll">
                        <!-- Cards injected here -->
                    </div>
                </div>
            </section>
        </main>

        <!-- ===== MANUAL SEARCH TAB ===== -->
        <main class="courtroom search-tab hidden" id="tab-search">
            <section class="court-panel search-panel" style="flex:1;">
                <div class="panel-title-bar">
                    <div class="title-bar-ornament left">━━◆</div>
                    <h2>MANUAL CITATION SEARCH</h2>
                    <div class="title-bar-ornament right">◆━━</div>
                </div>
                <div class="search-panel-body">
                    <p class="search-description">Enter any case name or citation to instantly verify it against the Supreme Court archive.</p>
                    <div class="search-input-group">
                        <div class="search-input-wrap">
                            <i class="fas fa-search search-icon"></i>
                            <input type="text" id="manual-search-input" 
                                placeholder="e.g. State of Bihar v. Ram Kumar Singh..." 
                                class="manual-search-input" 
                                onkeypress="if(event.key==='Enter') runManualSearch()">
                        </div>
                        <button class="search-btn" id="search-btn" onclick="runManualSearch()">
                            <i class="fas fa-gavel"></i> VERIFY
                        </button>
                    </div>
                    <div class="search-examples">
                        <span class="ex-label">Examples:</span>
                        <button class="ex-chip" onclick="fillSearch('Vashist Narayan Kumar v. State of Bihar')">Vashist Narayan Kumar v. State of Bihar</button>
                        <button class="ex-chip" onclick="fillSearch('Divya vs. Union of India')">Divya vs. Union of India</button>
                        <button class="ex-chip" onclick="fillSearch('Maneka Gandhi v. Union of India')">Maneka Gandhi v. Union of India</button>
                    </div>
                    <div class="search-results-area" id="search-results-area">
                        <div class="search-idle">
                            <i class="fas fa-balance-scale" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i>
                            <p>Enter a citation above to verify it</p>
                        </div>
                    </div>
                </div>
            </section>
        </main>

        <!-- ===== BULK UPLOAD TAB ===== -->
        <main class="courtroom bulk-tab hidden" id="tab-bulk">
            <section class="court-panel bulk-panel" style="flex:1;">
                <div class="panel-title-bar">
                    <div class="title-bar-ornament left">━━◆</div>
                    <h2>BULK DOCUMENT AUDIT</h2>
                    <div class="title-bar-ornament right">◆━━</div>
                </div>
                <div class="bulk-panel-body">
                    <p class="search-description">Upload multiple PDF documents at once for a combined citation audit.</p>
                    <div class="bulk-dropzone" id="bulk-dropzone" onclick="document.getElementById('bulk-file-input').click()">
                        <i class="fas fa-layer-group"></i>
                        <h3>DROP MULTIPLE PDFS HERE</h3>
                        <p>Or click to select files</p>
                        <input type="file" id="bulk-file-input" accept=".pdf" multiple hidden>
                    </div>
                    <div class="bulk-file-list" id="bulk-file-list"></div>
                    <button class="gavel-btn" id="bulk-audit-btn" disabled style="margin-top:1rem;" onclick="runBulkAudit()">
                        <div class="gavel-bg"></div>
                        <div class="gavel-content">
                            <span class="gavel-icon">📋</span>
                            <span class="gavel-text">AUDIT ALL DOCUMENTS</span>
                        </div>
                        <div class="gavel-shine"></div>
                    </button>
                    <div class="bulk-progress hidden" id="bulk-progress">
                        <div class="bulk-progress-bar"><div class="bulk-progress-fill" id="bulk-progress-fill"></div></div>
                        <div class="bulk-progress-text" id="bulk-progress-text">Processing...</div>
                    </div>
                    <div class="bulk-results-area" id="bulk-results-area"></div>
                </div>
            </section>
        </main>

        <!-- ===== HISTORY TAB ===== -->
        <main class="courtroom history-tab hidden" id="tab-history">
            <section class="court-panel history-panel" style="flex:1;">
                <div class="panel-title-bar">
                    <div class="title-bar-ornament left">━━◆</div>
                    <h2>AUDIT HISTORY</h2>
                    <div class="title-bar-ornament right">◆━━</div>
                </div>
                <div class="history-toolbar">
                    <span id="history-count" class="history-count">0 records</span>
                    <button class="action-btn" onclick="clearHistory()" style="background:rgba(220,50,50,0.15);border-color:rgba(220,50,50,0.3);color:#e87777;">
                        <i class="fas fa-trash"></i> Clear All
                    </button>
                </div>
                <div class="history-list" id="history-list">
                    <div class="history-empty">
                        <i class="fas fa-history" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i>
                        <p>No audit history yet. Run your first audit to see it here.</p>
                    </div>
                </div>
            </section>
        </main>

        <!-- ===== FOOTER / REGISTRY ===== -->
        <footer class="court-registry">
            <div class="registry-left">
                <span class="registry-badge">POWERED BY GROQ × LLAMA 3.3 70B</span>
            </div>
            <div class="registry-center">
                <span>यतो धर्मस्ततो जयः</span>
                <span class="registry-sep">•</span>
                <span>Where there is Righteousness, there is Victory</span>
            </div>
            <div class="registry-right">
                <span id="registry-records">—</span>
            </div>
        </footer>
    </div>

    <!-- ========== JUDGMENT ORDER MODAL ========== -->
    <div class="order-overlay hidden" id="order-overlay">
        <div class="order-sheet" id="order-sheet">
            <div class="order-header">
                <div class="order-header-ornament">
                    <span class="oh-line"></span>
                    <span class="oh-diamond">◆</span>
                    <span class="oh-line"></span>
                </div>
                <h3>JUDGMENT &amp; ORDER</h3>
                <div class="order-header-ornament">
                    <span class="oh-line"></span>
                    <span class="oh-diamond">◆</span>
                    <span class="oh-line"></span>
                </div>
                <button class="order-close" id="order-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="order-body" id="order-body">
                <!-- Populated dynamically -->
            </div>
            <div class="order-footer">
                <div class="order-seal">
                    <div class="stamp-circle">
                        <span>VERIFIED</span>
                    </div>
                </div>
                <p class="order-disclaimer">
                    This AI-generated audit is for informational purposes only and does not constitute legal advice.
                </p>
            </div>
        </div>
    </div>

    <!-- ========== AI SUMMARY MODAL ========== -->
    <div class="order-overlay hidden" id="summary-overlay">
        <div class="order-sheet">
            <div class="order-header">
                <div class="order-header-ornament">
                    <span class="oh-line"></span><span class="oh-diamond">◆</span><span class="oh-line"></span>
                </div>
                <h3>AI AUDIT SUMMARY</h3>
                <div class="order-header-ornament">
                    <span class="oh-line"></span><span class="oh-diamond">◆</span><span class="oh-line"></span>
                </div>
                <button class="order-close" onclick="document.getElementById('summary-overlay').classList.add('hidden')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="order-body" id="summary-body">
                <div class="summary-loading">
                    <i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i>
                    <p>Generating professional summary...</p>
                </div>
            </div>
            <div class="order-footer">
                <button class="action-btn export-pdf-btn" onclick="exportSummaryPDF()" style="margin:0 auto;">
                    <i class="fas fa-download"></i> Download Summary
                </button>
            </div>
        </div>
    </div>

    <!-- ========== TOAST CONTAINER ========== -->
    <div id="toast-container" class="toast-container"></div>

    <!-- ========== LEXAI CHATBOT ========== -->
    <div class="chat-bubble" id="chat-bubble" onclick="toggleChat()">
        <div class="chat-bubble-icon">⚖️</div>
        <div class="chat-bubble-label">LexAI</div>
        <div class="chat-notification" id="chat-notification" style="display:none;">1</div>
    </div>

    <div class="chat-pane hidden" id="chat-pane">
        <div class="chat-header">
            <div class="chat-header-left">
                <div class="chat-avatar">⚖️</div>
                <div>
                    <div class="chat-title">LexAI Legal Assistant</div>
                    <div class="chat-subtitle">Powered by Groq × Llama 3.3</div>
                </div>
            </div>
            <div class="chat-header-actions">
                <button title="Use audit context" onclick="toggleAuditContext()" id="ctx-btn" class="chat-action-btn">
                    <i class="fas fa-link"></i>
                </button>
                <button title="Clear chat" onclick="clearChat()" class="chat-action-btn">
                    <i class="fas fa-trash"></i>
                </button>
                <button onclick="toggleChat()" class="chat-action-btn">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
        <div class="chat-context-bar hidden" id="chat-context-bar">
            <i class="fas fa-link"></i> Audit context attached — LexAI will use your last audit results
        </div>
        <div class="chat-messages" id="chat-messages">
            <div class="chat-msg assistant">
                <div class="msg-avatar">⚖️</div>
                <div class="msg-bubble">
                    <p>Namaste! I'm <strong>LexAI</strong>, your AI legal assistant specializing in Indian law.</p>
                    <p style="margin-top:0.5rem;">I can help you:</p>
                    <ul style="margin:0.25rem 0 0 1rem;font-size:0.85rem;">
                        <li>Explain case law & legal concepts</li>
                        <li>Analyze your audit results</li>
                        <li>Research Indian constitutional law</li>
                        <li>Identify fabricated citations</li>
                    </ul>
                    <p style="margin-top:0.5rem;font-size:0.8rem;color:var(--text-secondary);">Run an audit first, then click the 🔗 button to give me context!</p>
                </div>
            </div>
        </div>
        <div class="chat-suggestions" id="chat-suggestions">
            <button class="chat-chip" onclick="sendSuggestion('What is a hallucinated citation?')">What is a hallucinated citation?</button>
            <button class="chat-chip" onclick="sendSuggestion('Explain Article 21 of the Indian Constitution')">Article 21</button>
            <button class="chat-chip" onclick="sendSuggestion('What does SCC stand for in Indian law?')">What is SCC?</button>
        </div>
        <div class="chat-input-area">
            <textarea id="chat-input" class="chat-input" placeholder="Ask any legal question..." rows="1"
                onkeypress="handleChatKey(event)" oninput="autoResizeChat(this)"></textarea>
            <button class="chat-send-btn" id="chat-send-btn" onclick="sendChatMessage()">
                <i class="fas fa-paper-plane"></i>
            </button>
        </div>
    </div>

    <!-- jsPDF for PDF export -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    <script src="/static/app.js"></script>
</body>
</html>
```

---



