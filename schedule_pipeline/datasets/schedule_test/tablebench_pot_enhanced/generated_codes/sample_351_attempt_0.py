import pandas as pd

df = pd.read_csv('table.csv')
# Convert '%' column to float for numerical comparison
df['%'] = pd.to_numeric(df['%'])
# Filter parties with more than 10% votes
parties_over_10_percent = df[df['%'] > 10]
# Count the number of such parties
count = len(parties_over_10_percent)
print(f"Final Answer: {count}")