import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'seats' column to integer type
df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
# Count how many elections had 75 or more seats
win_count = df[df['seats'] >= 75].shape[0]
print(f"Final Answer: {win_count}")