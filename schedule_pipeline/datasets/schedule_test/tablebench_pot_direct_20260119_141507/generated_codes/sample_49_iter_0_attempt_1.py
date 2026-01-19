import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the mean of the 'progressive conservative' column
avg_pc_votes = df['progressive conservative'].mean()
print(f"Final Answer: {avg_pc_votes:.1f}")