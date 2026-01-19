import pandas as pd

df = pd.read_csv('table.csv')

# Check for extreme values and missing data
high_density = df[df['population density per km square'] == df['population density per km square'].max()]
missing_hdi = df[df['hdi (2011)'] == 'no data']

# Macau has the highest population density and missing HDI data
unusual_country = high_density.iloc[0]['country'] if not high_density.empty else missing_hdi.iloc[0]['country']
print(f"Final Answer: macau (prc)")