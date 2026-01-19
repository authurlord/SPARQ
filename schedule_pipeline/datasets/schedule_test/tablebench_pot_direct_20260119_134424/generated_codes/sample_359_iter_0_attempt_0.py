import pandas as pd

df = pd.read_csv('table.csv')
# Convert '1 credit' column to numeric for comparison
df['1 credit'] = pd.to_numeric(df['1 credit'], errors='coerce')
# Count rows where '1 credit' >= 100
count = df[df['1 credit'] >= 100].shape[0]
print(f"Final Answer: {count}")