import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'average' column to numeric and calculate total sum
total_average = df['average'].astype(int).sum()
print(f"Final Answer: {total_average}")