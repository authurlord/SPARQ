import pandas as pd

df = pd.read_csv('table.csv')
# Convert population to integer for comparison
df['population'] = df['population'].astype(int)
# Count regions with population > 4 million
count_regions = df[df['population'] > 4000000].shape[0]
print(f"Final Answer: {count_regions}")