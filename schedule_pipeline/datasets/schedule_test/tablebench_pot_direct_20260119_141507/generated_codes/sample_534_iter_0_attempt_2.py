import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the data: remove percentage text and convert to numeric
def clean_numeric(value):
    # Extract the numeric part before parentheses or space
    match = re.search(r'(-?\d+\.?\d*)', str(value))
    if match:
        return float(match.group(1))
    return None

# Apply cleaning to GDP and area
df['gdp_billion'] = df['gdp (billion us)'].apply(clean_numeric)
df['area_km2'] = df['area (km square)'].apply(clean_numeric)

# Filter out rows with invalid values
valid_rows = df[(df['gdp_billion'].notna()) & (df['area_km2'].notna())]

# Calculate GDP per km²
valid_rows['gdp_per_km2'] = valid_rows['gdp_billion'] / valid_rows['area_km2']

# Find the country with the highest GDP per km²
max_gdp_per_km2 = valid_rows.loc[valid_rows['gdp_per_km2'].idxmax()]
highest_country = max_gdp_per_km2['member countries']

print(f"Final Answer: {highest_country}")