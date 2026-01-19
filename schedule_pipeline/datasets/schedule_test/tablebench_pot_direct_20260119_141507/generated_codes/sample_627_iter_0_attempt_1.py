import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract payouts for "natural royal flush" and "four of a kind" at 3 credits
natural_royal_flush_3_credit = df.loc[df['hand'] == 'natural royal flush', '3 credits'].values[0]
four_of_a_kind_3_credit = df.loc[df['hand'] == 'four of a kind', '3 credits'].values[0]

# Calculate total winnings for two separate games with 3-credit bets
total_winnings = (natural_royal_flush_3_credit + four_of_a_kind_3_credit) * 3

print(f"Final Answer: {total_winnings}")