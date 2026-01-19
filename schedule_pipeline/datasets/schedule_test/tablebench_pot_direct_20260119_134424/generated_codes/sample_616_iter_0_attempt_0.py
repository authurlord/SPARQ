import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'average' column for top 5 shows (first 5 rows)
top_5_average = df['average'].head(5).sum()
print(f"Final Answer: {top_5_average}")