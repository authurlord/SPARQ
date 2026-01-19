import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' column to numeric, treating '-' as NaN and filling with 0
df['total'] = pd.to_numeric(df['total'], errors='coerce').fillna(0).astype(int)
# Calculate total medals
total_medals = df['total'].sum()
print(f"Final Answer: {total_medals}")