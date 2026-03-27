"""
Offline pipeline to build the Case Intelligence Layer.
Run this ONCE (or whenever you add new PDFs).

Usage: python build_case_index.py
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import PyPDF2
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
OUTPUT_CARDS = Path("case_cards.parquet")
OUTPUT_EMBEDDINGS = Path("case_embeddings.npy")
OUTPUT_PATHS = Path("case_paths.json")  # Maps index → file path

# Rate limiting for Groq free tier
REQUESTS_PER_MINUTE = 28  # Stay under 30 RPM limit
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE

# If the pipeline crashes, it picks up where it left off
CHECKPOINT_FILE = Path("pipeline_checkpoint.json")
CHECKPOINT_INTERVAL = 50  # Save progress every 50 PDFs


def read_pdf(filepath: Path) -> str:
    """Extract text from PDF."""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return ""


def extract_strategic_text(full_text: str, front_chars=3000, back_chars=2000) -> str:
    """
    Extract the most informative parts of a judgment.
    
    WHY this works:
    - First ~3000 chars: Contains case name, parties, the legal question,
      and often the bench composition
    - Last ~2000 chars: Contains the HOLDING, the ORDER, and the disposition
      ("appeal allowed", "petition dismissed", etc.)
    
    This is FAR more useful than a random middle section, and keeps
    token usage manageable (~1500 tokens per call).
    """
    if len(full_text) <= front_chars + back_chars:
        return full_text
    
    front = full_text[:front_chars]
    back = full_text[-back_chars:]
    
    return f"{front}\n\n[... MIDDLE OF JUDGMENT OMITTED ...]\n\n{back}"


def extract_case_card(text: str, file_path: str) -> dict:
    """
    Use LLM to extract a structured case card from judgment text.
    
    This is the KEY innovation: instead of embedding raw judgment text
    (which is noisy and contains opposing arguments), we extract a 
    clean, structured summary that captures WHAT THE COURT ACTUALLY HELD.
    """
    prompt = f"""You are a legal research assistant. Read this excerpt from an Indian court judgment and extract a structured case card.

IMPORTANT INSTRUCTIONS:
1. Focus on what the COURT HELD, not what parties argued
2. Identify the core legal principle (ratio decidendi)
3. List the specific legal provisions interpreted
4. Write the searchable summary as if a lawyer would search for this case

Judgment excerpt:
{text}

Respond ONLY with this JSON structure:
{{
    "case_title": "Petitioner Name v. Respondent Name",
    "court": "Supreme Court of India" or "High Court of [State]",
    "legal_domain": "One of: Constitutional, Criminal, Civil, Tax, Labour, Environmental, Commercial, Family, Property, Administrative, Other",
    "key_statutes": ["Article 21", "Section 302 IPC", etc.],
    "core_legal_question": "One sentence describing the legal issue decided",
    "holding": "2-3 sentences on what the court actually held/decided",
    "key_principles": ["Principle 1 established", "Principle 2 established"],
    "searchable_summary": "A 3-4 sentence natural language summary combining the legal question, the holding, and the principles. Write this as if explaining to a lawyer what this case is useful for."
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "You extract structured legal metadata from court judgments. Output valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=800
        )
        
        card = json.loads(response.choices[0].message.content)
        card["source_file"] = str(file_path)
        return card
        
    except Exception as e:
        logger.error(f"LLM extraction failed for {file_path}: {e}")
        return {
            "case_title": "EXTRACTION_FAILED",
            "source_file": str(file_path),
            "searchable_summary": "",
            "error": str(e)
        }


def load_checkpoint() -> dict:
    """Load pipeline progress so we can resume after crashes."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_files": [], "cards": []}


def save_checkpoint(state: dict):
    """Save pipeline progress."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(state, f)


def discover_all_pdfs() -> list:
    """Find all judgment PDFs on disk."""
    all_pdfs = []
    for pdf_path in Path('.').rglob('*.pdf'):
        # Skip virtual environment and frontend files
        if any(skip in pdf_path.parts for skip in ['venv', 'frontend', '.git', 'node_modules']):
            continue
        all_pdfs.append(pdf_path)
    
    logger.info(f"Found {len(all_pdfs)} PDF files")
    return sorted(all_pdfs)


def build_index():
    """Main pipeline: Process all PDFs → Case Cards → Embeddings."""
    
    print("=" * 60)
    print("  CASE INTELLIGENCE LAYER - OFFLINE BUILDER")
    print("=" * 60)
    
    # 1. Discover all PDFs
    all_pdfs = discover_all_pdfs()
    all_pdfs = all_pdfs[:50]
    if not all_pdfs:
        print("❌ No PDFs found. Exiting.")
        return
    
    # 2. Load checkpoint (resume capability)
    state = load_checkpoint()
    processed_set = set(state["processed_files"])
    cards = state["cards"]
    
    remaining = [p for p in all_pdfs if str(p) not in processed_set]
    print(f"\n📊 Total PDFs: {len(all_pdfs)}")
    print(f"✅ Already processed: {len(processed_set)}")
    print(f"📝 Remaining: {len(remaining)}")
    print(f"⏱️  Estimated time: {len(remaining) * DELAY_BETWEEN_REQUESTS / 60:.1f} minutes")
    print()
    
    # 3. Process each PDF
    for i, pdf_path in enumerate(remaining):
        logger.info(f"[{i+1}/{len(remaining)}] Processing: {pdf_path.name}")
        
        # Extract text
        full_text = read_pdf(pdf_path)
        if not full_text.strip() or len(full_text) < 200:
            logger.warning(f"  ⚠️ Insufficient text, skipping")
            processed_set.add(str(pdf_path))
            continue
        
        # Get strategic excerpt
        excerpt = extract_strategic_text(full_text)
        
        # Extract case card via LLM
        card = extract_case_card(excerpt, str(pdf_path))
        cards.append(card)
        processed_set.add(str(pdf_path))
        
        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            state["processed_files"] = list(processed_set)
            state["cards"] = cards
            save_checkpoint(state)
            print(f"  💾 Checkpoint saved ({len(cards)} cards so far)")
        
        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # 4. Save final case cards
    print(f"\n📦 Saving {len(cards)} case cards...")
    cards_df = pd.DataFrame(cards)
    cards_df.to_parquet(OUTPUT_CARDS, index=False)
    print(f"  ✅ Saved to {OUTPUT_CARDS}")
    
    # 5. Build embeddings for the searchable_summary field
    print(f"\n🧠 Building embeddings...")
    summaries = cards_df['searchable_summary'].fillna('').tolist()
    
    # Filter out empty summaries
    valid_mask = [bool(s.strip()) for s in summaries]
    valid_summaries = [s for s, v in zip(summaries, valid_mask) if v]
    
    if valid_summaries:
        # Batch encode for efficiency
        embeddings = embedder.encode(
            valid_summaries, 
            show_progress_bar=True, 
            batch_size=64
        )
        
        # Create full embedding matrix (with zeros for failed extractions)
        full_embeddings = np.zeros((len(summaries), embeddings.shape[1]))
        valid_idx = 0
        for i, v in enumerate(valid_mask):
            if v:
                full_embeddings[i] = embeddings[valid_idx]
                valid_idx += 1
        
        np.save(OUTPUT_EMBEDDINGS, full_embeddings)
        print(f"  ✅ Saved embeddings to {OUTPUT_EMBEDDINGS}")
        print(f"  📐 Shape: {full_embeddings.shape}")
    
    # 6. Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    
    print(f"\n{'=' * 60}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"  📊 {len(cards)} case cards generated")
    print(f"  📊 {sum(valid_mask)} embeddings created")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    build_index()