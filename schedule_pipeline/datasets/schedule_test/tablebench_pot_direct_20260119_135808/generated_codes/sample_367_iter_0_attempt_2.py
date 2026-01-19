import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Number' column to numeric, non-numeric values become NaN
df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
# Count how many have Number > 1500
count = (df['Number'] > 1500).sum()
print(f"Final Answer: {count}")