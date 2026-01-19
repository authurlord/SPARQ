import pandas as pd

df = pd.read_csv('table.csv')
# Convert urban , % to numeric and find the first year where it exceeds 50%
df['urban , %'] = pd.to_numeric(df['urban , %'], errors='coerce')
for index, row in df.iterrows():
    if row['urban , %'] > 50:
        print(f"Final Answer: {row['year (january)']}")
        break