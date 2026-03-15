import pandas as pd
from pathlib import Path

def count_descriptions():
    print("🔍 Scanning all Parquet files to count descriptions...")
    
    parquet_files = [f for f in Path('.').rglob('*.parquet') if 'venv' not in f.parts]

    if not parquet_files:
        print("❌ No parquet files found.")
        return

    total_rows = 0
    total_with_desc = 0
    total_without_desc = 0
    files_processed = 0

    for file_to_load in parquet_files:
        try:
            df = pd.read_parquet(file_to_load)
            files_processed += 1
            
            # Count total rows in this file
            current_rows = len(df)
            total_rows += current_rows

            if 'description' not in df.columns:
                # If the column doesn't even exist, all rows are missing descriptions
                total_without_desc += current_rows
                continue

            # Convert to string and strip whitespace to catch sneaky empty strings
            # Replace NaNs with empty strings first
            desc_series = df['description'].fillna("").astype(str).str.strip()
            
            # Count rows where description is NOT empty and NOT "None Found" (based on your log)
            has_desc_mask = (desc_series != "") & (desc_series.str.lower() != "none found")
            
            with_desc = has_desc_mask.sum()
            without_desc = current_rows - with_desc

            total_with_desc += with_desc
            total_without_desc += without_desc

        except Exception as e:
            print(f"⚠️ Error reading {file_to_load}: {e}")

    # Print Final Report
    print("\n" + "="*50)
    print("📊 DATABASE DESCRIPTION REPORT")
    print("="*50)
    print(f"📁 Files Processed      : {files_processed}")
    print(f"📝 Total Cases (Rows)   : {total_rows:,}")
    print(f"✅ With Descriptions    : {total_with_desc:,} ({total_with_desc/total_rows*100:.1f}%)")
    print(f"❌ Without Descriptions : {total_without_desc:,} ({total_without_desc/total_rows*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    count_descriptions()