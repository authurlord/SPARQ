import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'semifinalists' to numeric, coercing errors to NaN
df['semifinalists'] = pd.to_numeric(df['semifinalists'], errors='coerce')
# Count how many countries have at least one semifinalist (value > 0)
count_semi = df[df['semifinalists'] > 0].shape[0]
print(f"Final Answer: {count_semi}")