import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter out rows where 'manchu' is not a number or percentage is invalid
df = df[df['manchu'].notna() & (df['percentage in manchu'] != '<0.01')]

# Convert 'manchu' and 'total population' to numeric for analysis
df['manchu'] = pd.to_numeric(df['manchu'], errors='coerce')
df['total population'] = pd.to_numeric(df['total population'], errors='coerce')

# Drop rows with NaN after conversion
df = df.dropna(subset=['manchu', 'total population'])

# Sort by Manchu population in descending order
df_sorted = df.sort_values(by='manchu', ascending=False)

# Display top 10 regions by Manchu population
print("Top 10 regions by Manchu population:")
print(df_sorted[['region', 'manchu']].head(10))

# Show regions with highest percentage of Manchu population
df_pct = df.sort_values(by='percentage in manchu', ascending=False)
print("\nRegions with highest percentage of Manchu population:")
print(df_pct[['region', 'percentage in manchu']].head(10))

# Final Answer: Summary of main components and insights
Final Answer: Liaoning, Hebei, Jilin, Northeast, Manchu concentration, high percentage in northern regions