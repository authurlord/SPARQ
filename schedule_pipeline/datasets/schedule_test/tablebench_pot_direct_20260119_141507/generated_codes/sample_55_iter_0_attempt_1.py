import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with 1 gold and 4 silver medals
result = df[(df['gold'] == 1) & (df['silver'] == 4)]['nation'].values[0]
print(f"Final Answer: {result}")