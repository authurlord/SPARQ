import pandas as pd

df = pd.read_csv('table.csv')
# Find the country with the highest average relative annual growth (%)
max_growth_country = df.loc[df['average relative annual growth (%)'].idxmax(), 'country (or dependent territory)']
print(f"Final Answer: {max_growth_country}")