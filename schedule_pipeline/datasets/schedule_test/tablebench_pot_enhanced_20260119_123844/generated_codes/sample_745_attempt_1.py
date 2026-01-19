import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where starts >= 5 to consider seasons with similar participation
filtered_df = df[df['starts'] >= 5]
# Calculate average finish position and total winnings
avg_finish = filtered_df['avg finish'].mean()
total_winnings = filtered_df['winnings'].astype(int).sum()

print(f"Final Answer: {avg_finish:.1f}, {total_winnings}")