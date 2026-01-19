import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'occurrence' column to integer
df['occurrence'] = pd.to_numeric(df['occurrence'])
# Filter rows where occurrence > 1 and count them
count = df[df['occurrence'] > 1].shape[0]
print(f"Final Answer: {count}")