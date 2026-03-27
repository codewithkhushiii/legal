import requests
import json
import PyPDF2

# 1. Change this to the name of ONE of your PDFs
TEST_PDF = "2023_1_1_230_EN.pdf"  

def extract_text(filepath):
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        return "".join([page.extract_text() or "" for page in reader.pages])

def test_ollama():
    print(f"Reading {TEST_PDF}...")
    text = extract_text(TEST_PDF)
    
    # Take a strategic chunk to fit in VRAM
    excerpt = text[:3000] + "\n\n[...]\n\n" + text[-2000:]
    
    prompt = f"""You are a legal research assistant. Extract a structured case card.
    
    IMPORTANT INSTRUCTIONS:
    1. Focus on what the COURT HELD, not what parties argued
    2. Identify the core legal principle (ratio decidendi)
    3. List the specific legal provisions interpreted
    4. Write the searchable summary as if a lawyer would search for this case

    Judgment excerpt:
    {excerpt}

    Respond ONLY with this JSON structure:
    {{
        "case_title": "Petitioner Name v. Respondent Name",
        "court": "Supreme Court of India" or "High Court of [State]",
        "legal_domain": "Domain",
        "key_statutes": ["Statute 1", "Statute 2"],
        "core_legal_question": "Question",
        "holding": "Holding",
        "key_principles": ["Principle 1", "Principle 2"],
        "searchable_summary": "Summary"
    }}"""

    print("Sending to Qwen2.5:3b (this might take a minute)...")
    
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096}
    })
    
    # Print the raw result
    result = response.json().get("message", {}).get("content", "")
    print("\n=== RESULT FROM OLLAMA ===")
    print(json.dumps(json.loads(result), indent=2))

if __name__ == "__main__":
    test_ollama()