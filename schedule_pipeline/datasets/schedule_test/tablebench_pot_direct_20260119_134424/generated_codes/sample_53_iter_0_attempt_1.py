import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where total medals are 13
nation_with_13_medals = df[df['total'] == '13']['nation'].iloc[0]
print(f"Final Answer: {nation_with_13_medals}")