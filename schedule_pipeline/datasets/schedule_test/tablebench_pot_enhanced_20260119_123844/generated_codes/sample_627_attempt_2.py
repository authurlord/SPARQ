import pandas as pd

df = pd.read_csv('table.csv')
# Find the payout for natural royal flush with 3 credits
natural_royal_flush_payout = int(df[df['hand'] == 'natural royal flush']['3 credits'].values[0])
# Find the payout for four of a kind with 3 credits
four_of_a_kind_payout = int(df[df['hand'] == 'four of a kind']['3 credits'].values[0])
# Calculate total winnings
total_winnings = natural_royal_flush_payout + four_of_a_kind_payout
print(f"Final Answer: {total_winnings}")