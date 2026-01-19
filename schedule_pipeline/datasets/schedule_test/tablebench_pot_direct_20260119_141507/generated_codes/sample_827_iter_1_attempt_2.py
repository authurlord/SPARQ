import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rank' to integer for proper comparison
df['rank'] = df['rank'].astype(int)

# Filter top 5 nations (rank <= 5)
top_5 = df[df['rank'] <= 5]
median_top_5 = top_5['total'].median()

# Median for all nations
median_all = df['total'].median()

print(f"Final Answer: {median_top_5}, {median_all}")