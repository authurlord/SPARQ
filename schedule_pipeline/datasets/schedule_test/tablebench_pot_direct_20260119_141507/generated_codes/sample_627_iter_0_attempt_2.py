import pandas as pd

df = pd.read_csv('table.csv')
# Extract payouts for "natural royal flush" and "four of a kind" at 3 credits
natural_royal_flush = df.loc[df['hand'] == 'natural royal flush', '3 credits'].values[0]
four_of_a_kind = df.loc[df['hand'] == 'four of a kind', '3 credits'].values[0]

# Bet amount
bet = 3

# Total winnings
total_winnings = (natural_royal_flush + four_of_a_kind) * bet
print(f"Final Answer: {total_winnings}")