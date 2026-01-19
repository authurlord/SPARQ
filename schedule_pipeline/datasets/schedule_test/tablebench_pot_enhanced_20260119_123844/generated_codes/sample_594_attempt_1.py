import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 singers by place (1st to 5th)
top_5 = df[df['place'].astype(int) <= 5]
# Sum the points
total_points = top_5['points'].sum()
print(f"Final Answer: {total_points}")