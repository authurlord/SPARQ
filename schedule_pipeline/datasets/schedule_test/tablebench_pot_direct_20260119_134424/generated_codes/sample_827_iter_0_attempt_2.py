import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 ranked nations
top_5_medals = df['total'].head(5)
# Calculate median for top 5
median_top_5 = top_5_medals.median()

# Calculate median for all nations
median_all = df['total'].median()

print(f"Final Answer: {median_top_5}, {median_all}")