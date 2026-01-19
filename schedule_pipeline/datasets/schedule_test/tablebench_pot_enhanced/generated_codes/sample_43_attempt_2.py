import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Himachal Pradesh
himachal_pradesh_plants = df[df['state'] == 'himachal pradesh']
# Convert capacity to numeric and sum
total_capacity = himachal_pradesh_plants['commissioned capacity (mw)'].astype(int).sum()
print(f"Final Answer: {total_capacity}")