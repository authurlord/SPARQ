import pandas as pd

df = pd.read_csv('table.csv')
# Convert '1 credit' column to numeric for comparison
df['1 credit'] = pd.to_numeric(df['1 credit'], errors='coerce')
# Filter rows where '1 credit' >= 100
filtered_rows = df[df['1 credit'] >= 100]
# Count the number of such rows
count = len(filtered_rows)
print(f"Final Answer: {count}")