import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the total row and find the nation with total = 13
nation_with_13_medals = df[df['total'] == 13]['nation'].values[0]
print(f"Final Answer: {nation_with_13_medals}")