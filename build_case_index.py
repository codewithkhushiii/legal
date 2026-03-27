"""
Optimized offline pipeline to build the Case Intelligence Layer.
Designed for low-VRAM GPUs (4GB) using HuggingFace transformers directly.

Usage: python build_case_index.py
"""

import os
import re
import json
import logging
import gc
import numpy as np
import pandas as pd
import PyPDF2
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
OUTPUT_CARDS       = Path("case_cards.parquet")
OUTPUT_EMBEDDINGS  = Path("case_embeddings.npy")
CHECKPOINT_FILE    = Path("pipeline_checkpoint.json")
CHECKPOINT_INTERVAL = 10

# ── HuggingFace Model Config ──
HF_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ── Tuning knobs for 4GB VRAM ──
MAX_NEW_TOKENS    = 512
MAX_INPUT_TOKENS  = 1024
FRONT_CHARS       = 1500
BACK_CHARS        = 1000
MAX_INPUT_CHARS   = 2000
PDF_READ_WORKERS  = 4
MIN_TEXT_LENGTH   = 500   # FIX #8: raised from 200 — avoids letting garbage through
MAX_PDFS          = None  # Set to an int to limit (e.g. 100); None = no limit
USE_4BIT          = True
USE_CPU_OFFLOAD   = False

# ─────────────────────────────────────────
# Lazy-loaded models
# ─────────────────────────────────────────
_embedder = None
_llm_model = None
_llm_tokenizer = None


def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


def get_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer

    logger.info(f"Loading LLM: {HF_MODEL_NAME}...")
    _llm_tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_NAME, trust_remote_code=True
    )
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    if USE_4BIT:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            logger.info("Using 4-bit quantization (bitsandbytes)")
        except ImportError:
            logger.warning("bitsandbytes not installed. Falling back to float16...")
            load_kwargs["device_map"] = "auto"
    elif USE_CPU_OFFLOAD:
        load_kwargs["device_map"] = "auto"
        load_kwargs["offload_folder"] = "offload"
    else:
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

    _llm_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME, **load_kwargs
    )
    _llm_model.eval()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory used by LLM: {allocated:.2f} GB")
    return _llm_model, _llm_tokenizer


def unload_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        del _llm_model
        _llm_model = None
    if _llm_tokenizer is not None:
        del _llm_tokenizer
        _llm_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("LLM unloaded from memory")


# ─────────────────────────────────────────
# PDF Reading
# ─────────────────────────────────────────
def read_pdf(filepath: Path) -> str:
    """
    FIX #6: Rewrote early-exit logic to avoid duplicating last-N pages.
    Previously, the last-4-pages append ran unconditionally after the break,
    causing duplicate content. Now tail pages are only collected when we
    actually truncate mid-document.
    """
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages = reader.pages
            total_pages = len(pages)
            target = FRONT_CHARS + BACK_CHARS + 500
            truncated = False

            front_chunks = []
            char_count = 0

            for page in pages:
                extracted = page.extract_text()
                if extracted:
                    front_chunks.append(extracted)
                    char_count += len(extracted)
                if char_count > target and total_pages > 20:
                    truncated = True
                    break

            if truncated:
                # Collect last 4 pages separately (they were not read above)
                tail_chunks = []
                for p in pages[-4:]:
                    ext = p.extract_text()
                    if ext:
                        tail_chunks.append(ext)
                return "\n".join(front_chunks) + "\n\n" + "\n".join(tail_chunks)

            return "\n".join(front_chunks)

    except Exception as e:
        logger.warning(f"Failed to read {filepath.name}: {e}")
        return ""


def read_pdfs_parallel(pdf_paths: list) -> dict:
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


def extract_strategic_text(full_text: str) -> str:
    text = full_text.strip()
    if not text:
        return ""
    if len(text) <= MAX_INPUT_CHARS:
        return text
    front = text[:FRONT_CHARS]
    back  = text[-BACK_CHARS:]
    combined = f"{front}\n\n[...]\n\n{back}"
    if len(combined) > MAX_INPUT_CHARS:
        combined = combined[:MAX_INPUT_CHARS]
    return combined


# ─────────────────────────────────────────
# LLM Extraction
# ─────────────────────────────────────────
SYSTEM_PROMPT = (
    "Extract structured legal metadata from Indian court judgments. "
    "Output valid JSON only."
)

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
    """
    FIX #5: Removed temperature and top_p when using greedy decoding
    (do_sample=False). Those parameters are ignored by the model in that
    mode and were misleading.

    FIX #7: Added a CPU fallback on GPU OOM — retries with a shorter
    input on CPU before giving up entirely.
    """
    model, tokenizer = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(text=text)}
    ]

    def _run_inference(mdl, tok, input_text, device):
        inputs = tok(
            input_text, return_tensors="pt",
            truncation=True, max_length=MAX_INPUT_TOKENS
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,           # FIX #5: greedy; no temperature/top_p needed
                repetition_penalty=1.1,
                pad_token_id=tok.eos_token_id
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    try:
        t0 = time()
        if hasattr(tokenizer, "apply_chat_template"):
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = (
                f"<|system|>{SYSTEM_PROMPT}<|end|>\n"
                f"<|user|>{USER_PROMPT_TEMPLATE.format(text=text)}<|end|>\n"
                f"<|assistant|>"
            )

        device = next(model.parameters()).device
        content = _run_inference(model, tokenizer, input_text, device)
        elapsed = time() - t0

        card = _parse_json_response(content)
        card["source_file"] = str(file_path)
        card["_llm_time_s"] = round(elapsed, 1)
        logger.info(f"  ✅ {Path(file_path).name} ({elapsed:.1f}s)")
        return card

    except torch.cuda.OutOfMemoryError:
        # FIX #7: Try again on CPU with a shorter excerpt before giving up
        logger.warning(
            f"  ⚠️ GPU OOM on {Path(file_path).name} — retrying on CPU with shorter input"
        )
        torch.cuda.empty_cache()
        gc.collect()
        try:
            short_text = text[:MAX_INPUT_CHARS // 2]
            short_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(text=short_text)}
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                input_text = tokenizer.apply_chat_template(
                    short_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                input_text = (
                    f"<|system|>{SYSTEM_PROMPT}<|end|>\n"
                    f"<|user|>{USER_PROMPT_TEMPLATE.format(text=short_text)}<|end|>\n"
                    f"<|assistant|>"
                )
            t0 = time()
            content = _run_inference(model, tokenizer, input_text, "cpu")
            elapsed = time() - t0
            card = _parse_json_response(content)
            card["source_file"] = str(file_path)
            card["_llm_time_s"] = round(elapsed, 1)
            card["_fallback"] = "cpu_oom_retry"
            logger.info(f"  ✅ {Path(file_path).name} (CPU fallback, {elapsed:.1f}s)")
            return card
        except Exception as e2:
            logger.error(f"  💥 CPU fallback also failed for {Path(file_path).name}: {e2}")
            return {
                "case_title": "EXTRACTION_FAILED",
                "source_file": str(file_path),
                "searchable_summary": "",
                "error": f"GPU OOM + CPU fallback failed: {e2}"
            }

    except Exception as e:
        logger.error(f"  ❌ Failed {Path(file_path).name}: {e}")
        return {
            "case_title": "EXTRACTION_FAILED",
            "source_file": str(file_path),
            "searchable_summary": "",
            "error": str(e)
        }


def _parse_json_response(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    logger.warning(f"  ⚠️ Could not parse JSON from LLM output")
    return {
        "case_title": "EXTRACTION_FAILED",
        "searchable_summary": "",
        "error": f"Unparseable response: {content[:200]}"
    }


# ─────────────────────────────────────────
# Checkpoint + Resume Management
# ─────────────────────────────────────────
def load_checkpoint() -> dict:
    """
    Resume state from (in priority order):
      1. Checkpoint file (interrupted run)
      2. Existing parquet (completed previous run)
      3. Fresh start
    """
    # ── Priority 1: In-progress checkpoint ──
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                state = json.load(f)
            logger.info(
                f"Resuming from checkpoint: "
                f"{len(state.get('processed_files', []))} files already done"
            )
            return state
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt checkpoint, checking parquet... ({e})")

    # ── Priority 2: Existing parquet from previous completed run ──
    if OUTPUT_CARDS.exists():
        try:
            existing_df = pd.read_parquet(OUTPUT_CARDS)
            processed_files = existing_df['source_file'].tolist()
            cards = existing_df.to_dict('records')
            # FIX #1: _llm_time_s is stripped before parquet save, so these
            # cards won't contribute to avg_llm. We note this explicitly rather
            # than silently returning 0 at the end.
            logger.info(
                f"Loaded {len(cards)} existing cards from {OUTPUT_CARDS} "
                f"(note: LLM timing data not available for pre-existing cards)"
            )
            return {
                "processed_files": processed_files,
                "cards": cards
            }
        except Exception as e:
            logger.warning(f"Could not load existing parquet: {e}")

    # ── Priority 3: Fresh start ──
    logger.info("Starting fresh — no checkpoint or parquet found")
    return {"processed_files": [], "cards": []}


def save_checkpoint(state: dict):
    """
    FIX #4: Only save processed_files list and a lightweight cards summary
    to the checkpoint to avoid serializing the full growing cards list on
    every batch. Full cards are reconstructed from the parquet on resume.

    Strategy: after each batch, flush successfully extracted cards to
    parquet (append/overwrite), and only store processed_files + failed
    entries in the checkpoint JSON.
    """
    tmp = CHECKPOINT_FILE.with_suffix('.tmp')
    # Separate failed cards (not in parquet) from successful ones
    failed_cards = [
        c for c in state.get("cards", [])
        if c.get("case_title") == "EXTRACTION_FAILED"
    ]
    checkpoint_data = {
        "processed_files": state["processed_files"],
        "failed_cards": failed_cards,   # Keep failures in JSON (small)
    }
    with open(tmp, 'w') as f:
        json.dump(checkpoint_data, f, default=str)
    tmp.replace(CHECKPOINT_FILE)


def flush_successful_cards_to_parquet(cards: list):
    """Write all successful cards accumulated so far to parquet."""
    successful = [c for c in cards if c.get("case_title") != "EXTRACTION_FAILED"]
    if not successful:
        return
    save_cards_to_parquet(successful, OUTPUT_CARDS)


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
# Parquet Save Helper
# ─────────────────────────────────────────
def save_cards_to_parquet(cards: list, output_path: Path):
    """Save cards with proper column type handling."""
    cards_df = pd.DataFrame(cards)

    string_cols = [
        'case_title', 'court', 'legal_domain', 'core_legal_question',
        'holding', 'searchable_summary', 'error', 'source_file'
    ]
    list_cols = ['key_statutes', 'key_principles']

    for col in string_cols:
        if col in cards_df.columns:
            cards_df[col] = cards_df[col].apply(
                lambda x: " ".join([str(i) for i in x]) if isinstance(x, list)
                else str(x) if pd.notna(x) else ""
            )

    for col in list_cols:
        if col in cards_df.columns:
            cards_df[col] = cards_df[col].apply(
                lambda x: [str(i) for i in x] if isinstance(x, list)
                else [str(x)] if isinstance(x, str) else []
            )

    # Drop internal timing column before saving
    drop_cols = [c for c in ['_llm_time_s', '_fallback'] if c in cards_df.columns]
    if drop_cols:
        cards_df = cards_df.drop(columns=drop_cols)

    cards_df.to_parquet(output_path, index=False)
    return cards_df


# ─────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────
def build_index():
    t_start = time()
    print("=" * 60)
    print("  CASE INTELLIGENCE LAYER — HUGGINGFACE DIRECT BUILDER")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  🖥️  Running on CPU (will be slow)")

    print(f"  🤖 Model: {HF_MODEL_NAME}")
    print(f"  🔧 4-bit quantization: {'ON' if USE_4BIT else 'OFF'}")

    # ── 1. Discover PDFs ──
    all_pdfs = discover_all_pdfs()

    # FIX #3: Removed hardcoded [:500] slice that silently overrode MAX_PDFS.
    # MAX_PDFS is now the single configurable limit.
    if MAX_PDFS:
        all_pdfs = all_pdfs[:MAX_PDFS]

    if not all_pdfs:
        print("❌ No PDFs found. Exiting.")
        return

    # ── 2. Resume from checkpoint OR existing parquet ──
    state = load_checkpoint()
    processed_set = set(state["processed_files"])
    cards = state.get("cards", [])

    # FIX #2: Use a single normalized set for all comparisons and updates.
    # Previously processed_set_normalized was built once but never updated,
    # making it inconsistent with processed_set after new files were added.
    def normalize_path(p) -> str:
        return str(Path(p))

    processed_normalized = {normalize_path(f) for f in processed_set}

    remaining = [
        p for p in all_pdfs
        if normalize_path(p) not in processed_normalized
    ]

    print(f"\n📊 Total PDFs: {len(all_pdfs)}")
    print(f"✅ Already processed: {len(processed_normalized)}")
    print(f"📝 Remaining: {len(remaining)}")
    print(f"⚙️  Config: input≤{MAX_INPUT_CHARS} chars, "
          f"output≤{MAX_NEW_TOKENS} tokens\n")

    if not remaining:
        print("✅ All PDFs already processed!")
        print("   Building embeddings from existing cards...")
    else:
        # ── 3. Pre-load the LLM ──
        logger.info("Pre-loading LLM...")
        get_llm()

        # ── 4. Process PDFs in batches ──
        BATCH_SIZE = 20
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start : batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

            print(f"\n── Batch {batch_num}/{total_batches} "
                  f"({len(batch)} PDFs) ──")

            logger.info("Reading PDFs in parallel...")
            pdf_texts = read_pdfs_parallel(batch)

            for pdf_path in batch:
                full_text = pdf_texts.get(pdf_path, "")
                if not full_text.strip() or len(full_text) < MIN_TEXT_LENGTH:
                    logger.warning(
                        f"  ⚠️ Skipping {pdf_path.name} (insufficient text)"
                    )
                    processed_normalized.add(normalize_path(pdf_path))
                    continue

                excerpt = extract_strategic_text(full_text)
                card = extract_case_card(excerpt, str(pdf_path))
                cards.append(card)
                processed_normalized.add(normalize_path(pdf_path))

            # FIX #4: Flush successful cards to parquet each batch;
            # checkpoint JSON only stores processed_files + failures (lightweight).
            flush_successful_cards_to_parquet(cards)
            state["processed_files"] = list(processed_normalized)
            state["cards"] = cards
            save_checkpoint(state)

            successful = sum(
                1 for c in cards
                if c.get("case_title") != "EXTRACTION_FAILED"
            )
            print(f"  💾 Checkpoint: {successful}/{len(cards)} successful cards")

        # ── Free LLM before loading embedder ──
        unload_llm()

    # ── 5. Save final case cards ──
    if not cards:
        print("❌ No cards extracted. Exiting.")
        return

    print(f"\n📦 Saving {len(cards)} case cards...")
    cards_df = save_cards_to_parquet(cards, OUTPUT_CARDS)
    print(f"  ✅ {OUTPUT_CARDS}")

    # ── 6. Build embeddings ──
    print(f"\n🧠 Building embeddings...")
    summaries = cards_df['searchable_summary'].fillna('').tolist()
    valid_mask = [bool(s.strip()) for s in summaries]
    valid_summaries = [s for s, v in zip(summaries, valid_mask) if v]

    if valid_summaries:
        embedder = get_embedder()
        embeddings = embedder.encode(
            valid_summaries,
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True
        )

        full_embeddings = np.zeros(
            (len(summaries), embeddings.shape[1]), dtype=np.float32
        )
        valid_idx = 0
        for i, v in enumerate(valid_mask):
            if v:
                full_embeddings[i] = embeddings[valid_idx]
                valid_idx += 1

        np.save(OUTPUT_EMBEDDINGS, full_embeddings)
        print(f"  ✅ {OUTPUT_EMBEDDINGS} "
              f"({full_embeddings.shape}, "
              f"{full_embeddings.nbytes/1024/1024:.1f} MB)")

    # ── 7. Cleanup checkpoint (only after everything saved) ──
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint file cleaned up")

    elapsed = time() - t_start
    successful = sum(
        1 for c in cards if c.get("case_title") != "EXTRACTION_FAILED"
    )
    failed = len(cards) - successful

    # FIX #1: Only average timing over cards that actually have it
    # (cards loaded from a previous parquet run won't have _llm_time_s).
    llm_times = [
        c.get("_llm_time_s") for c in cards
        if c.get("_llm_time_s") is not None
    ]
    avg_llm = np.mean(llm_times) if llm_times else 0
    timing_note = "" if llm_times else " (no timing data — resumed from parquet)"

    print(f"\n{'=' * 60}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"  📊 {successful} successful / {failed} failed / {len(cards)} total")
    print(f"  ⏱️  Total: {elapsed:.0f}s | Avg LLM call: {avg_llm:.1f}s{timing_note}")
    print(f"  📂 {OUTPUT_CARDS} + {OUTPUT_EMBEDDINGS}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    build_index()