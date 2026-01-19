import pandas as pd

df = pd.read_csv('table.csv')
# Convert '%' column to float for numerical comparison
df['%'] = pd.to_numeric(df['%'])
# Count parties with more than 10% votes
count_parties = (df['%'] > 10).sum()
print(f"Final Answer: {count_parties}")