import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'gdp per capita' to numeric, handling any potential non-numeric values
df['gdp per capita'] = pd.to_numeric(df['gdp per capita'], errors='coerce')
# Select top 20 countries by world rank
top_20 = df.head(20)
# Calculate median GDP per capita
median_gdp = top_20['gdp per capita'].median()
print(f"Final Answer: {median_gdp}")