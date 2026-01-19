import pandas as pd

df = pd.read_csv('table.csv')
# Convert the '%' column to float for numerical comparison
df['%'] = df['%'].astype(float)
# Count parties with more than 10% of votes
count_parties = df[df['%'] > 10].shape[0]
print(f"Final Answer: {count_parties}")