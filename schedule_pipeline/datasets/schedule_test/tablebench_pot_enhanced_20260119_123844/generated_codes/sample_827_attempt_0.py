import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' column to integer
df['total'] = pd.to_numeric(df['total'])

# Median for top 5 rankings
top_5_medals = df.head(5)['total'].median()

# Median for all countries
all_medals_median = df['total'].median()

print(f"Final Answer: {top_5_medals}, {all_medals_median}")