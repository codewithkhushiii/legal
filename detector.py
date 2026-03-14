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