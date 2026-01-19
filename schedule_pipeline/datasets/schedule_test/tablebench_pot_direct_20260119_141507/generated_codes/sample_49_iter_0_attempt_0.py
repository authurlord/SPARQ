import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average percentage of votes for the Progressive Conservative party
avg_progressive_conservative = df['progressive conservative'].mean()
print(f"Final Answer: {avg_progressive_conservative:.1f}")