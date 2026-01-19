import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'average' column to numeric
df['average'] = pd.to_numeric(df['average'])
# Calculate total average rating
total_average = df['average'].sum()
print(f"Final Answer: {total_average}")