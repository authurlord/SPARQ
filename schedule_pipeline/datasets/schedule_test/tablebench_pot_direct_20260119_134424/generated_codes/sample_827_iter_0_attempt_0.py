import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' column to integer for numerical operations
df['total'] = pd.to_numeric(df['total'])

# Median total medals for top 5 nations (ranks 1 to 5)
top_5_medals = df[df['rank'].astype(int) <= 5]['total']
median_top_5 = top_5_medals.median()

# Median total medals for all nations
median_all = df['total'].median()

print(f"Final Answer: {median_top_5}, {median_all}")