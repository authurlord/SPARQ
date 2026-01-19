import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'average' column for the top 5 shows
total_average_viewership = df['average'].astype(int).head(5).sum()
print(f"Final Answer: {total_average_viewership}")