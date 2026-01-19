import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'average' column for top 5 shows and sum them
total_average_viewership = df['average'].head(5).sum()
print(f"Final Answer: {total_average_viewership}")