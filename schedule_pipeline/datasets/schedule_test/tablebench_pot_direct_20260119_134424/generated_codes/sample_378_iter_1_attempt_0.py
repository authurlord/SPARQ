import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'seats' column to integer to allow numerical comparison
df['seats'] = pd.to_numeric(df['seats'])
# Count elections with 75 or more seats
count = (df['seats'] >= 75).sum()
print(f"Final Answer: {count}")