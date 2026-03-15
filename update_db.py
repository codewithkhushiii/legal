import pandas as pd
import re
from pathlib import Path

def extract_legal_provisions(text):
    """Hunts for Acts and Sections in messy OCR text."""
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
        
    # 3. Catch common abbreviations like IPC, CrPC, CPC
    if re.search(r'\bIPC\b', text): provisions.add("IPC")
    if re.search(r'\bCrPC\b', text): provisions.add("CrPC")
    if re.search(r'\bCPC\b', text): provisions.add("CPC")
        
    return ", ".join(sorted(list(provisions)))

def upgrade_all_parquets():
    print("🚀 Starting Database Upgrade: Mining provisions...")
    
    # Find all parquet files, ignoring the venv
    parquet_files = [f for f in Path('.').rglob('*.parquet') if 'venv' not in f.parts]
    
    if not parquet_files:
        print("❌ No .parquet files found.")
        return
        
    total_files = len(parquet_files)
    cases_upgraded = 0
    
    for i, file_path in enumerate(parquet_files, 1):
        print(f"📦 Processing file {i}/{total_files}: {file_path}")
        try:
            df = pd.read_parquet(file_path)
            
            if 'description' not in df.columns:
                print(f"   ⚠️ Skipping (no description column)")
                continue
                
            # Apply our regex hunter to the description column
            df['provisions'] = df['description'].apply(extract_legal_provisions)
            
            # Save it right back to the same file
            df.to_parquet(file_path)
            cases_upgraded += len(df)
            print(f"   ✅ Upgraded {len(df)} cases.")
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")

    print(f"\n🎉 SUCCESS! Upgraded {cases_upgraded} total cases with a new 'provisions' column.")

if __name__ == "__main__":
    upgrade_all_parquets()