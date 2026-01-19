import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with 18 gold medals
nation_with_18_gold = df[df['gold'] == 18]['nation'].values[0]
print(f"Final Answer: {nation_with_18_gold}")