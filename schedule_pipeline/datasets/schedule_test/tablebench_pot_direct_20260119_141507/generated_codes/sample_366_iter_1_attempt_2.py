import pandas as pd

df = pd.read_csv('table.csv')
# Convert population to integer
df['population'] = df['population'].str.replace(',', '').astype(int)
# Count regions with population > 4 million
count = df[df['population'] > 4000000].shape[0]
print(f"Final Answer: {count}")