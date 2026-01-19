import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where hand is 'four of a kind , 2 - 4'
payout_row = df[df['hand'] == 'four of a kind , 2 - 4']
# Get the payout for 3 credits
payout = payout_row['3 credits'].values[0]
print(f"Final Answer: {payout}")