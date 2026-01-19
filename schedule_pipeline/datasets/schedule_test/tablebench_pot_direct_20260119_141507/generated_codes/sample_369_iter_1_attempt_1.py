import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'semifinalists' to integer type to avoid string comparison errors
df['semifinalists'] = pd.to_numeric(df['semifinalists'], errors='coerce')
# Count how many countries have at least one semifinalist
count_semi = df[df['semifinalists'] >= 1].shape[0]
print(f"Final Answer: {count_semi}")