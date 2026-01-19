import pandas as pd
import numpy as np

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert land area to numeric (already numeric in string form)
df['land area (km 2 )'] = pd.to_numeric(df['land area (km 2 )'], errors='coerce')
df['rainfall by depth (mm / year)'] = pd.to_numeric(df['rainfall by depth (mm / year)'], errors='coerce')
df['rainfall by volume (km 3 / year)'] = pd.to_numeric(df['rainfall by volume (km 3 / year)'], errors='coerce')
df['surface run off (km 3 / year)'] = pd.to_numeric(df['surface run off (km 3 / year)'], errors='coerce')
df['infiltration (km 3 / year)'] = pd.to_numeric(df['infiltration (km 3 / year)'], errors='coerce')
df['evapotranspiration (km 3 / year)'] = pd.to_numeric(df['evapotranspiration (km 3 / year)'], errors='coerce')

# Select only relevant columns for correlation
corr_data = df[['land area (km 2 )', 'rainfall by depth (mm / year)', 'rainfall by volume (km 3 / year)', 
                'surface run off (km 3 / year)', 'infiltration (km 3 / year)', 'evapotranspiration (km 3 / year)']]

# Compute correlation matrix
corr_matrix = corr_data.corr()

# Get the correlation with 'rainfall by volume (km 3 / year)'
correlations = corr_matrix['rainfall by volume (km 3 / year)'].dropna()

# Sort by absolute value and get top 2
top_two_factors = correlations.abs().sort_values(ascending=False).head(2).index.tolist()

# Final Answer: Top two factors influencing rainfall by volume
print(f"Final Answer: {top_two_factors[0]}, {top_two_factors[1]}")