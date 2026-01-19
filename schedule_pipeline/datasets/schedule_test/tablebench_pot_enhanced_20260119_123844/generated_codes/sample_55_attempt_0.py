import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nation with 1 gold and 4 silver medals
result = df[(df['gold'] == '1') & (df['silver'] == '4')]
nation = result['nation'].values[0]
print(f"Final Answer: {nation}")