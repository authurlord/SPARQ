import pandas as pd

df = pd.read_csv('table.csv')

# Calculate average finish position across all years
avg_finish = df['avg finish'].astype(float).mean()

# Calculate total winnings across all years (sum of winnings)
total_winnings = df['winnings'].astype(int).sum()

print(f"Final Answer: {avg_finish:.1f}, {total_winnings}")