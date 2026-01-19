import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'semifinalists' column to integer type to avoid comparison errors
df['semifinalists'] = pd.to_numeric(df['semifinalists'], errors='coerce')
# Count countries with at least one semifinalist
count_countries = (df['semifinalists'] > 0).sum()
print(f"Final Answer: {count_countries}")