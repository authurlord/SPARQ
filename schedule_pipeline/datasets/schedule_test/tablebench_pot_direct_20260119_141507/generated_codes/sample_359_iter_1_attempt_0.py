import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where '1 credit' is 100 or more
payouts_1_credit = df['1 credit'].astype(int)
count_100_or_more = (payouts_1_credit >= 100).sum()
print(f"Final Answer: {count_100_or_more}")