import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the last row (theoretical return)
df_filtered = df.iloc[:-1]
# Convert '1 credit' column to integers
df_filtered['1 credit'] = df_filtered['1 credit'].astype(int)
# Count how many have payout >= 100
count = (df_filtered['1 credit'] >= 100).sum()
print(f"Final Answer: {count}")