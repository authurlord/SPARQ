import pandas as pd

df = pd.read_csv('table.csv')
# Replace '-' with 0 and convert to integer
df['total'] = df['total'].replace('-', 0).astype(int)
# Calculate total medals
total_medals = df['total'].sum()
print(f"Final Answer: {total_medals}")