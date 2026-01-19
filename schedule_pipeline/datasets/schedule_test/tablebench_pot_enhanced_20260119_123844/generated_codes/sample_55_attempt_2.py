import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where gold is 1 and silver is 4
result = df[(df['gold'] == '1') & (df['silver'] == '4')]
nation = result['nation'].iloc[0]
print(f"Final Answer: {nation}")