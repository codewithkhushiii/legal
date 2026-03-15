import pandas as pd
import PyPDF2
import re
import os
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress PyPDF2 warnings about messy PDFs
warnings.filterwarnings("ignore", category=UserWarning)

def extract_legal_provisions(text):
    """Hunts for Acts and Sections in the raw PDF text."""
    if not isinstance(text, str) or not text.strip():
        return ""
        
    provisions = set()
    
    # 1. Catch "Sections X to Y" or "Section X"
    section_pattern = r'(?:Section|Sections|Sec\.)\s*(\d+(?:\s*to\s*\d+)?|[A-Z\d]+)'
    for match in re.findall(section_pattern, text, re.IGNORECASE):
        provisions.add(f"Section {match.strip()}")
        
    # 2. Catch "[Name] Act, [Year]"
    act_pattern = r'([A-Z][A-Za-z\s]+Act,\s*\d{4})'
    for match in re.findall(act_pattern, text):
        provisions.add(match.strip())
        
    # 3. Catch common abbreviations
    if re.search(r'\bIPC\b', text): provisions.add("IPC")
    if re.search(r'\bCrPC\b', text): provisions.add("CrPC")
    if re.search(r'\bCPC\b', text): provisions.add("CPC")
        
    return ", ".join(sorted(list(provisions)))

def find_pdf_path(row):
    """Attempts to locate the PDF file on disk using multiple strategies."""
    # Strategy 1: The 'path' column
    if 'path' in row and pd.notna(row['path']):
        candidate = Path(str(row['path']))
        if candidate.exists():
            return candidate

    # Strategy 2: The 'case_id' or 'nc_display' column (e.g. 2024_1_1_10)
    file_id = None
    if 'nc_display' in row and pd.notna(row['nc_display']):
        file_id = str(row['nc_display']).strip()
    elif 'case_id' in row and pd.notna(row['case_id']):
        file_id = str(row['case_id']).strip()

    if file_id:
        year_match = re.match(r'^(\d{4})', file_id)
        if year_match:
            year = year_match.group(1)
            possible_folders = [f"judgement_{year}", f"judgment_{year}"]
            sub_folder = f"extracted_{year}_case"
            
            for base in possible_folders:
                candidate = Path(base) / sub_folder / f"{file_id}.pdf"
                if candidate.exists():
                    return candidate
    return None

def process_single_pdf(pdf_path, max_pages=3):
    """Reads ONLY the first few pages of a PDF to save extreme amounts of time."""
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
        extracted_text = ""
        # Read up to max_pages, or the length of the document, whichever is smaller
        pages_to_read = min(max_pages, len(reader.pages))
        
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                extracted_text += page_text + "\n"
                
        return extract_legal_provisions(extracted_text)
    except Exception:
        # Silently fail on corrupted PDFs so the script doesn't crash
        return ""

def mine_all_pdfs():
    print("🚀 Starting Deep PDF Mining Operation...")
    
    parquet_files = [f for f in Path('.').rglob('*.parquet') if 'venv' not in f.parts]
    
    if not parquet_files:
        print("❌ No Parquet files found.")
        return

    total_provisions_found = 0

    for file_to_load in parquet_files:
        print(f"\n📂 Loading: {file_to_load}")
        try:
            df = pd.read_parquet(file_to_load)
            
            # Create the column if it doesn't exist yet
            if 'provisions' not in df.columns:
                df['provisions'] = ""

            # Use tqdm to show a progress bar for the rows in THIS specific file
            tqdm.pandas(desc="Mining PDFs")
            
            # Apply our massive extractor function row by row
            def process_row(row):
                # If we already found provisions previously, skip it to save time
                if pd.notna(row.get('provisions')) and str(row.get('provisions')).strip():
                    return row['provisions']
                    
                pdf_path = find_pdf_path(row)
                if pdf_path:
                    return process_single_pdf(pdf_path, max_pages=3)
                return ""

            df['provisions'] = df.progress_apply(process_row, axis=1)
            
            # Count how many we actually found in this batch
            found_count = df['provisions'].apply(lambda x: 1 if str(x).strip() else 0).sum()
            total_provisions_found += found_count
            
            # Overwrite the parquet file with the newly enriched data
            df.to_parquet(file_to_load)
            print(f"✅ Saved. {found_count} cases now have legal provisions in this file.")

        except Exception as e:
            print(f"⚠️ Error processing file {file_to_load}: {e}")

    print("\n" + "="*50)
    print("🎉 MINING COMPLETE!")
    print(f"Total cases enriched with provisions: {total_provisions_found}")
    print("="*50)

if __name__ == "__main__":
    mine_all_pdfs()