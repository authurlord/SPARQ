import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'percentage in manchu population' to numeric, coercing errors to NaN
df['percentage in manchu population'] = pd.to_numeric(df['percentage in manchu population'], errors='coerce')

# Filter rows where percentage > 5
filtered_df = df[df['percentage in manchu population'] > 5]

# Sum the 'total population' of those filtered regions
total_population = filtered_df['total population'].sum()

print(f"Final Answer: {total_population}")