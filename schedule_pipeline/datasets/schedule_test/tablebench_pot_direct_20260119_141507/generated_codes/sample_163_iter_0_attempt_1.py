import pandas as pd

df = pd.read_csv('table.csv')
# Convert all relevant columns to numeric
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')
df['profits (billion )'] = pd.to_numeric(df['profits (billion )'], errors='coerce')
df['assets (billion )'] = pd.to_numeric(df['assets (billion )'], errors='coerce')
df['market value (billion )'] = pd.to_numeric(df['market value (billion )'], errors='coerce')

# Calculate correlation with market value
correlations = {
    'sales (billion )': df['sales (billion )'].corr(df['market value (billion )']),
    'profits (billion )': df['profits (billion )'].corr(df['market value (billion )']),
    'assets (billion )': df['assets (billion )'].corr(df['market value (billion )'])
}

# Find the variable with the highest absolute correlation
max_corr_var = max(correlations, key=lambda x: abs(correlations[x]))
print(f"Final Answer: {max_corr_var}")