import pandas as pd

df = pd.read_csv('table.csv')
# Convert pop density to numeric, coercing errors to NaN if any
df['pop density (per km square)'] = pd.to_numeric(df['pop density (per km square)'], errors='coerce')
# Count how many have population density > 3000
count_high_density = df[df['pop density (per km square)'] > 3000].shape[0]
print(f"Final Answer: {count_high_density}")