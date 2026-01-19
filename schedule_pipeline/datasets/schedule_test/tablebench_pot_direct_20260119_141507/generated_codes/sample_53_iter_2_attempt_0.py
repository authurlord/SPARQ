import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where total is 13 and get the nation
nation_with_13_medals = df[df['total'] == 13]['nation'].values[0]
print(f"Final Answer: {nation_with_13_medals}")