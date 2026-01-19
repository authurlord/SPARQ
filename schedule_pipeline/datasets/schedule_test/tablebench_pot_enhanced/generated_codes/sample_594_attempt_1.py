import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 singers by place (1 to 5)
top_5_singers = df[df['place'].astype(int) <= 5]
total_points = top_5_singers['points'].sum()
print(f"Final Answer: {total_points}")