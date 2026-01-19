import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'average' column and sum the top 5 values
top_5_average = df['average'].head(5).sum()
print(f"Final Answer: {top_5_average}")