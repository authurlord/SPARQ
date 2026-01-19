import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where player name is "alpha ambrose kargbo"
alpha_ambrose_kargbo = df[df['player name'] == 'alpha ambrose kargbo']
# Get the number of 3-pointers
three_pointers = alpha_ambrose_kargbo['3 - points'].values[0]
print(f"Final Answer: {three_pointers}")