import pandas as pd

df = pd.read_csv('table.csv')
# Convert international passengers to numeric (they are already numeric strings)
df['international passengers'] = pd.to_numeric(df['international passengers'], errors='coerce')

# Drop any rows with missing values due to conversion
df = df.dropna(subset=['international passengers'])

# Calculate correlation between international passengers and rank
correlation = df['international passengers'].corr(df['rank'])

print(f"Final Answer: {correlation:.2f}")