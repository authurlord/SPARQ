import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where state is 'himachal pradesh'
himachal_plants = df[df['state'] == 'himachal pradesh']
# Convert 'commissioned capacity (mw)' to numeric and sum
total_capacity = himachal_plants['commissioned capacity (mw)'].astype(int).sum()
print(f"Final Answer: {total_capacity}")