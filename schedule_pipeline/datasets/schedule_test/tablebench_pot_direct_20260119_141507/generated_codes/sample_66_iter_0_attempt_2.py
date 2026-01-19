import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where player name is "Alpha Ambrose Kargbo"
alpha_ambrose_kargbo = df[df['player name'] == 'alpha ambrose kargbo']
# Extract the 3-pointers value
three_pointers = alpha_ambrose_kargbo['3 - points'].values[0]
print(f"Final Answer: {three_pointers}")