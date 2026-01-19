import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Alpha Ambrose Kargbo
kargbo_row = df[df['player name'] == 'alpha ambrose kargbo']
# Extract the 3-pointers made
three_pointers = kargbo_row['3 - points'].values[0]
print(f"Final Answer: {three_pointers}")