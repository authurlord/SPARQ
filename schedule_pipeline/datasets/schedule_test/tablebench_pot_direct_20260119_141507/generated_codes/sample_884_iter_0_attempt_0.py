import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert to numeric for calculations
df['total s ton'] = pd.to_numeric(df['total s ton'], errors='coerce')
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'], errors='coerce')
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'], errors='coerce')

# Sort by year to ensure chronological order
df = df.sort_values('year').reset_index(drop=True)

# Calculate year-over-year percentage increase
df['domestic_inc'] = ((df['domestic s ton'] - df['domestic s ton'].shift(1)) / df['domestic s ton'].shift(1)) * 100
df['foreign_inc'] = ((df['foreign total s ton'] - df['foreign total s ton'].shift(1)) / df['foreign total s ton'].shift(1)) * 100

# Find year where domestic increase > foreign increase
result_year = df[(df['domestic_inc'] > df['foreign_inc']) & (df['domestic_inc'].notna()) & (df['foreign_inc'].notna())]['year'].iloc[0] if not df[(df['domestic_inc'] > df['foreign_inc']) & (df['domestic_inc'].notna()) & (df['foreign_inc'].notna())].empty else None

print(f"Final Answer: {result_year}")