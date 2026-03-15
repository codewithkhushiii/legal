import pandas as pd
from pathlib import Path

def explore_descriptions():
    print("🔍 Searching for Parquet files...")
    
    # Find all parquet files, ignoring the virtual environment
    parquet_files = [f for f in Path('.').rglob('*.parquet') if 'venv' not in f.parts]

    if not parquet_files:
        print("❌ No parquet files found in the current directory.")
        return

    # Just grab the first file we find for a quick peek
    file_to_load = parquet_files[0]
    print(f"📂 Loading data from: {file_to_load}\n")
    
    try:
        df = pd.read_parquet(file_to_load)
        
        if 'description' not in df.columns:
            print("❌ 'description' column not found in this dataset.")
            print(f"Available columns are: {list(df.columns)}")
            return

        # Drop NaNs or completely empty strings so we don't just print blank spaces
        valid_descriptions = df['description'].dropna()
        valid_descriptions = valid_descriptions[valid_descriptions.str.strip() != ""]
        
        if valid_descriptions.empty:
            print("⚠️ The 'description' column exists, but it's completely empty in this file!")
            return

        print(f"✅ Found {len(valid_descriptions)} valid descriptions. Here are the top 5:\n")
        
        # Grab the top 5 and print them cleanly
        top_5 = valid_descriptions.head(5).tolist()
        
        print(top_5)

        for i, desc in enumerate(top_5, 1):
            print(f"--- 📄 Sample {i} ---")
            # Truncate at 1000 characters just in case it's a massive wall of text
            text_to_print = desc if len(desc) < 1000 else desc[:1000] + "\n...[TRUNCATED]"
            print(text_to_print)
            print("-" * 60 + "\n")

    except Exception as e:
        print(f"⚠️ Error reading file: {e}")

if __name__ == "__main__":
    explore_descriptions()