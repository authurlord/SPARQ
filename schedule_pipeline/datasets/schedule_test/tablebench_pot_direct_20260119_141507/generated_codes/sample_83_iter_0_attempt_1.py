import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation where total is 6
nation_with_6_medals = df[df['total'] == 6]['nation'].values[0]
print(f"Final Answer: {nation_with_6_medals}")