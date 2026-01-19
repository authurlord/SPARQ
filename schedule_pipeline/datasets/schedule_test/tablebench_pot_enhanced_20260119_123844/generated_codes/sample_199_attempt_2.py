import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'deaths' to numeric, coercing errors to NaN
df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce')

# Drop rows where deaths or major hurricanes are missing
df_clean = df.dropna(subset=['number of major hurricanes', 'deaths'])

# Calculate correlation
correlation = df_clean['number of major hurricanes'].corr(df_clean['deaths'])

print(f"Final Answer: {correlation:.2f}")