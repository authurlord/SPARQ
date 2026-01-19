import pandas as pd

df = pd.read_csv('table.csv')
# Find the country with the highest agricultural use percentage
max_agricultural_country = df.loc[df['agricultural use'].idxmax(), 'Unnamed: 0']
print(f"Final Answer: {max_agricultural_country}")