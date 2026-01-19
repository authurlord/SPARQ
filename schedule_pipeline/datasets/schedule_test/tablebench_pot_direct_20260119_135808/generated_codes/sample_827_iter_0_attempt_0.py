import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 nations by rank
top_5 = df[df['rank'].astype(int) <= 5]
median_top_5 = top_5['total'].median()
median_all = df['total'].median()
print(f"Final Answer: {median_top_5}, {median_all}")