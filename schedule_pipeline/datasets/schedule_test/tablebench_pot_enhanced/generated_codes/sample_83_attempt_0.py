import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where total medals are 6
nation_with_6_medals = df[df['total'] == '6']['nation'].iloc[0]
print(f"Final Answer: {nation_with_6_medals}")