import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'change (%)' to numeric, handling any non-numeric values
df['change (%)'] = pd.to_numeric(df['change (%)'], errors='coerce')

# Drop rows with missing values after conversion
df = df.dropna(subset=['change (%)'])

# Select numerical columns for correlation
numerical_cols = ['area (km square)', 'population density', 'change (%)']
df_numeric = df[numerical_cols]

# Compute correlation matrix
correlation = df_numeric.corr()

# Check if 'area (km square)' or 'population density' has a significant correlation (> 0.3 or < -0.3)
area_corr = correlation.loc['change (%)', 'area (km square)']
density_corr = correlation.loc['change (%)', 'population density']

# Determine if either has a significant influence
if abs(area_corr) >= 0.3 or abs(density_corr) >= 0.3:
    influence = f"area (km square): {area_corr:.3f}, population density: {density_corr:.3f}"
else:
    influence = 'no clear impact'

print(f"Final Answer: {influence}")