import pandas as pd

df = pd.read_csv('table.csv')
# Convert '1 credit' column to numeric for comparison
df['1 credit'] = pd.to_numeric(df['1 credit'], errors='coerce')
# Filter rows where payout for 1 credit is 100 or more
high_payout_hands = df[df['1 credit'] >= 100]
# Count the number of such hands
count = len(high_payout_hands)
print(f"Final Answer: {count}")