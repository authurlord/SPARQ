import pandas as pd

df = pd.read_csv('table.csv')
# Calculate average finish position and total winnings over the period
avg_finish = df['avg finish'].mean()
total_winnings = df['winnings'].mean()

print(f"Final Answer: {avg_finish:.1f}, {total_winnings:.0f}")