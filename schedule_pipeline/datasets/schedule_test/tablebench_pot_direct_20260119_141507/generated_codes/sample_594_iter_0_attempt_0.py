import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 singers by place (place <= 5)
top_5_points = df[df['place'] <= 5]['points'].sum()
print(f"Final Answer: {top_5_points}")