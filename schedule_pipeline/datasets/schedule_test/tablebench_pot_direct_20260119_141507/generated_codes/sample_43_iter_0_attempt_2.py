import pandas as pd

df = pd.read_csv('table.csv')
# Filter power plants in Himachal Pradesh and sum their commissioned capacity
total_capacity_hp = df[df['state'] == 'himachal pradesh']['commissioned capacity (mw)'].sum()
print(f"Final Answer: {total_capacity_hp}")