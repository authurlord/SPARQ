import pandas as pd

df = pd.read_csv('table.csv')

# Filter top 5 nations (rank 1 to 5)
top_5 = df[df['rank'].between(1, 5)]
median_top_5 = top_5['total'].median()

# Median for all countries
median_all = df['total'].median()

print(f"Final Answer: {median_top_5}, {median_all}")