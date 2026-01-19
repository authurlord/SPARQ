import pandas as pd

df = pd.read_csv('table.csv')
# Calculate average finish and total winnings across all years
avg_finish = df['avg finish'].mean()
total_winnings = df['winnings'].sum()

print(f"Final Answer: {avg_finish:.1f}, {total_winnings}")