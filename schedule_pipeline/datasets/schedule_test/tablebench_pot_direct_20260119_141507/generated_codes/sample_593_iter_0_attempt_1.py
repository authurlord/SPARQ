import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 2008 and 2009, then sum the 'wins' column
total_wins = df[(df['year'] == '2008') | (df['year'] == '2009')]['wins'].sum()
print(f"Final Answer: {total_wins}")