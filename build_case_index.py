"""
Optimized offline pipeline to build the Case Intelligence Layer.
Designed for low-VRAM GPUs (4GB) running Ollama.

Usage: python build_case_index.py
"""

import os
import re
import json
import logging
import requests
import numpy as np
import pandas as pd
import PyPDF2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from time import time, sleep

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
OUTPUT_CARDS       = Path("case_cards_ollama.parquet")
OUTPUT_EMBEDDINGS  = Path("case_embeddings_ollama.npy")
CHECKPOINT_FILE    = Path("pipeline_checkpoint_ollama.json")
CHECKPOINT_INTERVAL = 10        # Save more often for safety

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL_NAME  = "qwen2.5:3b"

# ── Tuning knobs for 4GB VRAM ──
NUM_CTX           = 2048        # ⬇ from 4096 — halves VRAM for context
FRONT_CHARS       = 1500        # ⬇ from 3000 — less input = faster inference
BACK_CHARS        = 1000        # ⬇ from 2000
MAX_INPUT_CHARS   = 2000        # Hard cap on what we send to LLM
PDF_READ_WORKERS  = 4           # Parallel PDF text extraction
LLM_TIMEOUT       = 90          # Seconds before we abandon a call
LLM_RETRIES       = 2           # Retry on failure
MIN_TEXT_LENGTH    = 200         # Skip near-empty PDFs
MAX_PDFS           = None       # Set to int to limit, None for all

# ─────────────────────────────────────────
# Lazy-loaded embedding model
# ─────────────────────────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model (one-time)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


# ─────────────────────────────────────────
# PDF Reading (parallelizable, no GPU)
# ─────────────────────────────────────────
def read_pdf(filepath: Path) -> str:
    """Extract text from PDF — optimized to stop early."""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            chunks = []
            char_count = 0
            target = FRONT_CHARS + BACK_CHARS + 500  # slight buffer

            # Read only as many pages as we need for the front
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    chunks.append(extracted)
                    char_count += len(extracted)
                # Once we have enough for front+back, stop reading middle pages
                # But we still need the LAST pages, so read all if short doc
                if char_count > target and len(reader.pages) > 20:
                    # Grab last few pages for the "back" portion
                    for p in reader.pages[-4:]:
                        ext = p.extract_text()
                        if ext:
                            chunks.append(ext)
                    break

            return "\n".join(chunks)
    except Exception as e:
        logger.warning(f"Failed to read {filepath.name}: {e}")
        return ""


def read_pdfs_parallel(pdf_paths: list) -> dict:
    """Read multiple PDFs in parallel (I/O bound, no GPU)."""
    results = {}
    with ThreadPoolExecutor(max_workers=PDF_READ_WORKERS) as executor:
        future_to_path = {
            executor.submit(read_pdf, p): p for p in pdf_paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[path] = future.result()
            except Exception as e:
                logger.warning(f"PDF read error {path.name}: {e}")
                results[path] = ""
    return results


# ─────────────────────────────────────────
# Strategic Text Extraction
# ─────────────────────────────────────────
def extract_strategic_text(full_text: str) -> str:
    """
    Extract the most informative parts, capped to MAX_INPUT_CHARS.
    Court judgments: header has parties/statutes, tail has the holding.
    """
    text = full_text.strip()
    if not text:
        return ""

    if len(text) <= MAX_INPUT_CHARS:
        return text

    front = text[:FRONT_CHARS]
    back  = text[-BACK_CHARS:]

    combined = f"{front}\n\n[...]\n\n{back}"

    # Hard cap — if still too long, truncate
    if len(combined) > MAX_INPUT_CHARS:
        combined = combined[:MAX_INPUT_CHARS]

    return combined


# ─────────────────────────────────────────
# LLM Extraction (Ollama) — Leaner Prompt
# ─────────────────────────────────────────

# Shorter prompt = fewer tokens for the model to process = faster
SYSTEM_PROMPT = "Extract structured legal metadata from Indian court judgments. Output valid JSON only."

USER_PROMPT_TEMPLATE = """Extract a case card from this Indian judgment excerpt. 
Focus on what the COURT HELD.

Excerpt:
{text}

Return JSON:
{{
  "case_title": "X v. Y",
  "court": "court name",
  "legal_domain": "Constitutional|Criminal|Civil|Tax|Labour|Environmental|Commercial|Family|Property|Administrative|Other",
  "key_statutes": ["statutes"],
  "core_legal_question": "one sentence",
  "holding": "2-3 sentences",
  "key_principles": ["principles"],
  "searchable_summary": "3-4 sentence summary for lawyer search"
}}"""


def extract_case_card(text: str, file_path: str) -> dict:
    """Call Ollama with retry logic and tight timeouts."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(text=text)}
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": NUM_CTX,        # Key VRAM saver
            "num_predict": 512,        # ← Cap output tokens (cards are short)
        }
    }

    for attempt in range(1, LLM_RETRIES + 1):
        try:
            t0 = time()
            response = requests.post(
                OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT
            )
            response.raise_for_status()
            elapsed = time() - t0

            content = response.json().get("message", {}).get("content", "{}")
            card = json.loads(content)
            card["source_file"] = str(file_path)
            card["_llm_time_s"] = round(elapsed, 1)

            logger.info(f"  ✅ {Path(file_path).name} ({elapsed:.1f}s)")
            return card

        except requests.exceptions.Timeout:
            logger.warning(
                f"  ⏱️ Timeout attempt {attempt}/{LLM_RETRIES} for {Path(file_path).name}"
            )
            if attempt < LLM_RETRIES:
                sleep(2)

        except json.JSONDecodeError as e:
            logger.warning(f"  ⚠️ Bad JSON from LLM for {Path(file_path).name}: {e}")
            # Try to salvage partial JSON
            try:
                # Sometimes LLM outputs markdown-wrapped JSON
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    card = json.loads(match.group())
                    card["source_file"] = str(file_path)
                    card["_llm_time_s"] = round(time() - t0, 1)
                    return card
            except:
                pass

        except Exception as e:
            logger.error(f"  ❌ Attempt {attempt}/{LLM_RETRIES} failed: {e}")
            if attempt < LLM_RETRIES:
                sleep(2)

    return {
        "case_title": "EXTRACTION_FAILED",
        "source_file": str(file_path),
        "searchable_summary": "",
        "error": "All retries exhausted"
    }


# ─────────────────────────────────────────
# Checkpoint Management
# ─────────────────────────────────────────
def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_files": [], "cards": []}


def save_checkpoint(state: dict):
    # Write to temp file first, then rename (atomic on most OS)
    tmp = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(state, f)
    tmp.replace(CHECKPOINT_FILE)


# ─────────────────────────────────────────
# PDF Discovery
# ─────────────────────────────────────────
SKIP_DIRS = {'venv', 'frontend', '.git', 'node_modules', '__pycache__', '.venv'}

def discover_all_pdfs() -> list:
    all_pdfs = []
    for pdf_path in Path('.').rglob('*.pdf'):
        if SKIP_DIRS & set(pdf_path.parts):
            continue
        all_pdfs.append(pdf_path)
    logger.info(f"Found {len(all_pdfs)} PDF files")
    return sorted(all_pdfs)


# ─────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────
def build_index():
    t_start = time()
    print("=" * 60)
    print("  CASE INTELLIGENCE LAYER — OPTIMIZED OLLAMA BUILDER")
    print("=" * 60)

    # ── 1. Discover PDFs ──
    all_pdfs = discover_all_pdfs()
    all_pdfs = all_pdfs[:5]
    if MAX_PDFS:
        all_pdfs = all_pdfs[:MAX_PDFS]
    if not all_pdfs:
        print("❌ No PDFs found. Exiting.")
        return

    # ── 2. Resume from checkpoint ──
    state = load_checkpoint()
    processed_set = set(state["processed_files"])
    cards = state["cards"]

    remaining = [p for p in all_pdfs if str(p) not in processed_set]
    print(f"\n📊 Total PDFs: {len(all_pdfs)}")
    print(f"✅ Already processed: {len(processed_set)}")
    print(f"📝 Remaining: {len(remaining)}")
    print(f"⚙️  Config: num_ctx={NUM_CTX}, input≤{MAX_INPUT_CHARS} chars, "
          f"output≤512 tokens\n")

    if not remaining:
        print("Nothing to process. Building embeddings from existing cards...")
    else:
        # ── 3. Pre-read PDFs in parallel batches ──
        BATCH_SIZE = 20  # Read 20 PDFs at a time in parallel
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start : batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

            print(f"\n── Batch {batch_num}/{total_batches} "
                  f"({len(batch)} PDFs) ──")

            # Parallel PDF reading
            logger.info("Reading PDFs in parallel...")
            pdf_texts = read_pdfs_parallel(batch)

            # Sequential LLM calls (GPU is the bottleneck, can't parallelize)
            for pdf_path in batch:
                full_text = pdf_texts.get(pdf_path, "")
                if not full_text.strip() or len(full_text) < MIN_TEXT_LENGTH:
                    logger.warning(f"  ⚠️ Skipping {pdf_path.name} (insufficient text)")
                    processed_set.add(str(pdf_path))
                    continue

                excerpt = extract_strategic_text(full_text)
                card = extract_case_card(excerpt, str(pdf_path))
                cards.append(card)
                processed_set.add(str(pdf_path))

            # Checkpoint after every batch
            state["processed_files"] = list(processed_set)
            state["cards"] = cards
            save_checkpoint(state)
            
            successful = sum(1 for c in cards if c.get("case_title") != "EXTRACTION_FAILED")
            print(f"  💾 Checkpoint: {successful}/{len(cards)} successful cards")

    # ── 4. Save case cards ──
    if not cards:
        print("❌ No cards extracted. Exiting.")
        return

    print(f"\n📦 Saving {len(cards)} case cards...")
    cards_df = pd.DataFrame(cards)
    # cards_df.to_parquet(OUTPUT_CARDS, index=False)
    
    string_cols = ['case_title', 'court', 'legal_domain', 'core_legal_question', 'holding', 'searchable_summary', 'error', 'source_file']
    list_cols = ['key_statutes', 'key_principles']
    
    for col in string_cols:
        if col in cards_df.columns:
            # Convert accidental lists to joined strings, and handle NaNs
            cards_df[col] = cards_df[col].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x) if pd.notna(x) else "")
            
    for col in list_cols:
        if col in cards_df.columns:
            # Convert accidental strings to lists
            cards_df[col] = cards_df[col].apply(lambda x: [x] if isinstance(x, str) else x if isinstance(x, list) else [])
    # -------------------------------------------------------

    cards_df.to_parquet(OUTPUT_CARDS, index=False)
    print(f"  ✅ {OUTPUT_CARDS}")

    # ── 5. Build embeddings ──
    print(f"\n🧠 Building embeddings...")
    summaries = cards_df['searchable_summary'].fillna('').tolist()
    valid_mask = [bool(s.strip()) for s in summaries]
    valid_summaries = [s for s, v in zip(summaries, valid_mask) if v]

    if valid_summaries:
        embedder = get_embedder()
        embeddings = embedder.encode(
            valid_summaries,
            show_progress_bar=True,
            batch_size=128,       # ⬆ from 64 — MiniLM is tiny, can handle it
            normalize_embeddings=True  # Pre-normalize for cosine similarity
        )

        full_embeddings = np.zeros((len(summaries), embeddings.shape[1]),
                                    dtype=np.float32)  # float32 not float64
        valid_idx = 0
        for i, v in enumerate(valid_mask):
            if v:
                full_embeddings[i] = embeddings[valid_idx]
                valid_idx += 1

        np.save(OUTPUT_EMBEDDINGS, full_embeddings)
        print(f"  ✅ {OUTPUT_EMBEDDINGS} "
              f"({full_embeddings.shape}, {full_embeddings.nbytes/1024/1024:.1f} MB)")

    # ── 6. Cleanup ──
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    elapsed = time() - t_start
    successful = sum(1 for c in cards if c.get("case_title") != "EXTRACTION_FAILED")
    failed = len(cards) - successful

    # Timing stats
    llm_times = [c.get("_llm_time_s", 0) for c in cards if c.get("_llm_time_s")]
    avg_llm = np.mean(llm_times) if llm_times else 0

    print(f"\n{'=' * 60}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"  📊 {successful} successful / {failed} failed / {len(cards)} total")
    print(f"  ⏱️  Total: {elapsed:.0f}s | Avg LLM call: {avg_llm:.1f}s")
    print(f"  📂 {OUTPUT_CARDS} + {OUTPUT_EMBEDDINGS}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    build_index()