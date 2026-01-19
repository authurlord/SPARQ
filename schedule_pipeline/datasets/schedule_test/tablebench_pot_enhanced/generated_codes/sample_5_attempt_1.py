import pandas as pd

df = pd.read_csv('table.csv')
# Clean the '2001 general' column: extract numeric value before parentheses if present
df['2001 general'] = df['2001 general'].str.replace(r'\s*\(.*\)', '', regex=True)
# Convert to float and compute mean
mean_2001_general = df['2001 general'].astype(float).mean()
print(f"Final Answer: {mean_2001_general:.1f}")