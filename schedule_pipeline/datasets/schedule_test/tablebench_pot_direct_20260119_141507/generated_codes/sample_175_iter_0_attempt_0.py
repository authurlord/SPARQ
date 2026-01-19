import pandas as pd

df = pd.read_csv('table.csv')

# Convert land area to float
df['land area (km 2 )'] = pd.to_numeric(df['land area (km 2 )'], errors='coerce')

# Select only numeric columns for correlation
numeric_columns = ['land area (km 2 )', 'rainfall by depth (mm / year)', 'surface run off (km 3 / year)', 
                   'infiltration (km 3 / year)', 'evapotranspiration (km 3 / year)', 'rainfall by volume (km 3 / year)']

df_numeric = df[numeric_columns]

# Compute correlation with 'rainfall by volume (km 3 / year)'
correlations = df_numeric.corr()['rainfall by volume (km 3 / year)'].dropna()

# Get top 2 factors by absolute correlation
top_2_factors = correlations.abs().sort_values(ascending=False).head(2).index.tolist()

print(f"Final Answer: {top_2_factors[0]}, {top_2_factors[1]}")