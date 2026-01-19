import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'international passengers' to numeric (handle any formatting issues)
df['international passengers'] = pd.to_numeric(df['international passengers'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['international passengers', 'rank'])

# Convert 'rank' to numeric (it's already numeric)
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

# Calculate correlation between international passengers and rank
correlation = df['international passengers'].corr(df['rank'])

print(f"Final Answer: {correlation:.3f}")