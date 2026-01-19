import pandas as pd

df = pd.read_csv('table.csv')
# Filter for position 37th in 20 km walk event
result = df[(df['Position'] == '37th') & (df['Event'] == '20 km walk')]['Competition'].values[0]
print(f"Final Answer: {result}")