import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where gold medals are 18
nation_with_18_gold = df[df['gold'] == '18']['nation'].iloc[0]
print(f"Final Answer: {nation_with_18_gold}")