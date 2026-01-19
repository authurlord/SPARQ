import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bp comp 1 (˚C)' and '% wt comp 1' to numeric, handling negative signs and whitespace
df['bp comp 1 (˚C)'] = pd.to_numeric(df['bp comp 1 (˚C)'], errors='coerce')
df['% wt comp 1'] = pd.to_numeric(df['% wt comp 1'], errors='coerce')

# Calculate correlation coefficient
correlation = df['bp comp 1 (˚C)'].corr(df['% wt comp 1'])

print(f"Final Answer: {correlation:.4f}")