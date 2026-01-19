import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for "four of a kind , 2 - 4"
payout_row = df[df['hand'] == 'four of a kind , 2 - 4']
# Extract the payout for 3 credits
payout_3_credits = payout_row['3 credits'].values[0]
print(f"Final Answer: {payout_3_credits}")