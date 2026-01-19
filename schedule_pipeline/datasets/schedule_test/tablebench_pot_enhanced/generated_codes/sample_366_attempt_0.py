import pandas as pd

df = pd.read_csv('table.csv')
# Convert population column to integer
df['population'] = pd.to_numeric(df['population'])
# Count regions with population > 4 million
count_regions = df[df['population'] > 4000000].shape[0]
print(f"Final Answer: {count_regions}")