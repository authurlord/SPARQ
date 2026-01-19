import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Sort by 'gdp world rank' to get top 20 countries
df_sorted = df.sort_values(by='gdp world rank').head(20)
# Convert 'gdp per capita' to numeric, coercing errors to NaN, then drop NaN
df_sorted['gdp per capita'] = pd.to_numeric(df_sorted['gdp per capita'], errors='coerce')
# Drop rows with NaN values (invalid entries like 'n / a')
df_sorted = df_sorted.dropna(subset=['gdp per capita'])
# Calculate median of GDP per capita
median_gdp = np.median(df_sorted['gdp per capita'])
print(f"Final Answer: {median_gdp:.0f}")