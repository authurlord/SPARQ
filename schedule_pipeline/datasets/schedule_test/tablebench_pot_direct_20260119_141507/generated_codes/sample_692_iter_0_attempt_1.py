import pandas as pd

df = pd.read_csv('table.csv')
# Find the country with the highest average relative annual growth (%)
max_growth_row = df[df['average relative annual growth (%)'] == df['average relative annual growth (%)'].max()]
country_with_max_growth = max_growth_row.iloc[0]['country (or dependent territory)']
print(f"Final Answer: {country_with_max_growth}")