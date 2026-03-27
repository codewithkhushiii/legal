import pandas as pd

# Load the saved LLM extractions
df = pd.read_parquet('case_cards.parquet')

# Use pandas' native to_json, which handles the numpy arrays automatically!
print(df.head(2).to_json(orient='records', indent=4))