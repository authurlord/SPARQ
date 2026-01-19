import pandas as pd

df = pd.read_csv('table.csv')
# Select the 'average' column and sum the top 5 values
total_average_viewership = df['average'].head(5).sum()
print(f"Final Answer: {total_average_viewership}")