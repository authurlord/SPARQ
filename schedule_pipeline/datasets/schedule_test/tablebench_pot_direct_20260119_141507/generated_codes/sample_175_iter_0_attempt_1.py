import pandas as pd

df = pd.read_csv('table.csv')

# Select only the relevant columns for correlation
columns_to_analyze = ['rainfall by depth (mm / year)', 'surface run off (km 3 / year)', 
                      'infiltration (km 3 / year)', 'evapotranspiration (km 3 / year)']
target_column = 'rainfall by volume (km 3 / year)'

# Convert all columns to numeric (handling any potential parsing issues)
df_numeric = df[columns_to_analyze + [target_column]].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN due to conversion issues
df_clean = df_numeric.dropna()

# Compute correlation with target column
correlations = df_clean.corr()[target_column].sort_values(ascending=False)

# Get top 2 factors
top_2_factors = correlations.head(2).index.tolist()

print(f"Final Answer: {top_2_factors[0]}, {top_2_factors[1]}")