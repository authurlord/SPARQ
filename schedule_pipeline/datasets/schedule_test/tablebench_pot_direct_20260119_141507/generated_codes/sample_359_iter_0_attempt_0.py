import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where '1 credit' is 100 or more and count them
count_100_or_more = df[df['1 credit'].astype(int) >= 100].shape[0]
print(f"Final Answer: {count_100_or_more}")