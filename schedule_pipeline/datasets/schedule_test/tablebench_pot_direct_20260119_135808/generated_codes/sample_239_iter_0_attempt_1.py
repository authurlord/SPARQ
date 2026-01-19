import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['typhoid fever'] = pd.to_numeric(df['typhoid fever'], errors='coerce')

# Calculate correlation
correlation = df['typhus'].corr(df['typhoid fever'])

print(f"Final Answer: {correlation:.2f}")