import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Australia and Russia
australia_total = df[df['nation'] == 'australia']['total'].values[0]
russia_total = df[df['nation'] == 'russia']['total'].values[0]

# Compare and determine the higher total
if australia_total > russia_total:
    winner = 'australia'
else:
    winner = 'russia'

print(f"Final Answer: {winner}")