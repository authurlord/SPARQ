import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'place' to get top 5 performers
top_5 = df.sort_values(by='place').head(5)
# Sum the points of the top 5
total_points = top_5['points'].sum()
print(f"Final Answer: {total_points}")