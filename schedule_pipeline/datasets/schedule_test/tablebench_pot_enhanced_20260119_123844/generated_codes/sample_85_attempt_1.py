import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where hand is 'four of a kind , 2 - 4'
payout = df[df['hand'] == 'four of a kind , 2 - 4']['3 credits'].values[0]
print(f"Final Answer: {payout}")