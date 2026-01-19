import pandas as pd

df = pd.read_csv('table.csv')
# Find the country with the highest agricultural use percentage
max_agricultural_use = df['agricultural use'].idxmax()
country_with_max_agricultural_use = df.loc[max_agricultural_use, 'Unnamed: 0']
print(f"Final Answer: {country_with_max_agricultural_use}")