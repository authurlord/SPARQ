import pandas as pd

df = pd.read_csv('table.csv')
# Convert '%' column to float
df['%'] = pd.to_numeric(df['%'])
# Count parties with more than 10% votes
count_parties = df[df['%'] > 10.0].shape[0]
print(f"Final Answer: {count_parties}")