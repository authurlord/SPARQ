import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'place' ascending to get top 5 by place (lowest place number)
top_5 = df.sort_values(by='place').head(5)
# Sum the points of the top 5 singers
total_points = top_5['points'].sum()
print(f"Final Answer: {total_points}")