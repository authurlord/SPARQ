import pandas as pd

df = pd.read_csv('table.csv')
# Convert '%' column to float for numerical comparison
df['%'] = df['%'].astype(float)
# Count parties with more than 10% of votes
count_parties = (df['%'] > 10).sum()
print(f"Final Answer: {count_parties}")