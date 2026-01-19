import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where winnings is 411728
year = df[df['winnings'] == 411728]['year'].values[0]
print(f"Final Answer: {year}")