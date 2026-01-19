import pandas as pd

df = pd.read_csv('table.csv')

# Drop non-numeric columns for correlation (only 'viewers (in millions)' is numeric)
# Convert 'viewers (in millions)' to numeric
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'], errors='coerce')

# Select numeric columns for correlation
numeric_cols = ['season', 'season premiere', 'season finale', 'tv season', 'rank', 'viewers (in millions)']
df_numeric = df[numeric_cols]

# Drop rows with missing values in viewers
df_clean = df_numeric.dropna(subset=['viewers (in millions)'])

# Compute correlation with viewers (in millions)
correlations = df_clean.corr()['viewers (in millions)'].abs()

# Find columns with correlation > 0.3 (threshold for significant influence)
significant_factors = correlations[correlations > 0.3].index.tolist()

if len(significant_factors) == 0:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {', '.join(significant_factors)}")