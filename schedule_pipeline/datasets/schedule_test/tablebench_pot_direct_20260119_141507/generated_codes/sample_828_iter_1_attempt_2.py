import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where 'gdp per capita' is "n / a"
df_clean = df[df['gdp per capita'] != 'n / a']

# Convert 'gdp per capita' to numeric, coercing errors to NaN
df_clean['gdp per capita'] = pd.to_numeric(df_clean['gdp per capita'], errors='coerce')

# Sort by 'gdp world rank' to get top 20 countries by world rank
df_top_20 = df_clean.sort_values(by='gdp world rank').head(20)

# Calculate the median of the GDP per capita for these 20 countries
median_gdp = df_top_20['gdp per capita'].median()
print(f"Final Answer: {median_gdp:.0f}")