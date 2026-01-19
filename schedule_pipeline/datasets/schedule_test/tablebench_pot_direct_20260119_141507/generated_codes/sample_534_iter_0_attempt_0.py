import pandas as pd
import re

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: convert 'population', 'area (km square)', 'gdp (billion us)' to numeric
# Remove percentage signs and parse numbers
def clean_numeric(x):
    if isinstance(x, str):
        # Extract numeric part from string (e.g., "256762167 (+ 33.41%)" -> 256762167)
        match = re.search(r'(-?\d+\.?\d*)', x)
        if match:
            return float(match.group(1))
    return None

# Apply cleaning to relevant columns
df['population'] = df['population'].apply(clean_numeric)
df['area (km square)'] = df['area (km square)'].apply(clean_numeric)
df['gdp (billion us)'] = df['gdp (billion us)'].apply(clean_numeric)

# Drop rows where any of the required columns is NaN (invalid data)
df_clean = df.dropna(subset=['population', 'area (km square)', 'gdp (billion us)'])

# Calculate GDP per square kilometer
df_clean['gdp_per_km2'] = df_clean['gdp (billion us)'] / df_clean['area (km square)']

# Find the country with the highest GDP per km²
max_gdp_per_km2_row = df_clean.loc[df_clean['gdp_per_km2'].idxmax()]
highest_country = max_gdp_per_km2_row['member countries']

print(f"Final Answer: {highest_country}")