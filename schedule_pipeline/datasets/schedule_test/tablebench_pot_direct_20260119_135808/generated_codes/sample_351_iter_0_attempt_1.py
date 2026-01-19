import pandas as pd

df = pd.read_csv('table.csv')
# Convert '%' column to float
df['%'] = pd.to_numeric(df['%'])
# Filter parties with more than 10% votes
parties_above_10 = df[df['%'] > 10]
# Count the number of such parties
count_parties = len(parties_above_10)
print(f"Final Answer: {count_parties}")