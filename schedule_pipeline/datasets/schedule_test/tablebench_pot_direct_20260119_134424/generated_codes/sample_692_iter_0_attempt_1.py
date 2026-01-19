import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum 'average relative annual growth (%)'
max_growth_row = df.loc[df['average relative annual growth (%)'].idxmax()]
country_with_max_growth = max_growth_row['country (or dependent territory)']
print(f"Final Answer: {country_with_max_growth}")