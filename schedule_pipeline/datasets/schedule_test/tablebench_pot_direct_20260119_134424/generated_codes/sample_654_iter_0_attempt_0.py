import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'IP' and 'SO' to numeric, handling non-numeric values if any
df['IP'] = pd.to_numeric(df['IP'], errors='coerce')
df['SO'] = pd.to_numeric(df['SO'], errors='coerce')

# Drop rows with missing values in IP or SO
df.dropna(subset=['IP', 'SO'], inplace=True)

# Calculate correlation coefficient
correlation = df['IP'].corr(df['SO'])

print(f"Final Answer: {correlation:.3f}")