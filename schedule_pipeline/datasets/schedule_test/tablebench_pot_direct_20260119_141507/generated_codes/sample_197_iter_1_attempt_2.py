import pandas as pd

df = pd.read_csv('table.csv')
# Correct column names based on actual data
arrivals_col = 'internl tourist arrivals 2011 (x1000)'
receipts_col = 'internl tourism receipts 2011 (million usd )'

# Drop rows with 'n / d' values to avoid NaN issues
df_clean = df.dropna(subset=[arrivals_col, receipts_col])

# Calculate correlation between tourist arrivals and receipts
correlation = df_clean[arrivals_col].corr(df_clean[receipts_col])
print(f"Final Answer: {correlation:.2f}")