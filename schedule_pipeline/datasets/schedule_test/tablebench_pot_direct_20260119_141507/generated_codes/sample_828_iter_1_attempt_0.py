import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter top 20 countries by world rank (world rank from 1 to 20)
top_20 = df[df['gdp world rank'].between(1, 20)]

# Convert 'gdp per capita' to numeric, handling non-numeric values
top_20['gdp per capita'] = pd.to_numeric(top_20['gdp per capita'], errors='coerce')

# Drop rows with NaN (invalid GDP values)
top_20_clean = top_20.dropna(subset=['gdp per capita'])

# Compute median GDP per capita
median_gdp = top_20_clean['gdp per capita'].median()

print(f"Final Answer: {median_gdp:.0f}")