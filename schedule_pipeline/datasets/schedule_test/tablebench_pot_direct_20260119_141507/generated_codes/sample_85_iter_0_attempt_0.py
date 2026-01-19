import pandas as pd

df = pd.read_csv('table.csv')
# Find the payout for "four of a kind, 2-4" when betting 3 credits
payout = df.loc[df['hand'] == 'four of a kind , 2 - 4', '3 credits'].values[0]
print(f"Final Answer: {payout}")