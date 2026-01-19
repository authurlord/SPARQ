import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'speakers' column to numeric, coercing errors to NaN
df['speakers'] = pd.to_numeric(df['speakers'], errors='coerce')

# Remove rows where speakers is NaN (invalid entries)
df = df.dropna(subset=['speakers'])

# Basic insights: describe the distribution of speakers
print(f"Mean speakers per council area: {df['speakers'].mean():.0f}")
print(f"Median speakers per council area: {df['speakers'].median():.0f}")
print(f"Max speakers: {df['speakers'].max():.0f}")
print(f"Min speakers: {df['speakers'].min():.0f}")

# Final Answer: Mean, Median, Max, Min
Final Answer: Mean, Median, Max, Min