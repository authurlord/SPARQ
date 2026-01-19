import pandas as pd

df = pd.read_csv('table.csv')
# Find the country with the highest agricultural use percentage
max_agricultural = df['agricultural use'].idxmax()
country_with_max_agricultural = df.loc[max_agricultural, 'Unnamed: 0']
print(f"Final Answer: {country_with_max_agricultural}")