import pandas as pd

df = pd.read_csv('table.csv')

# Median total for top 5 nations (rank <= 5)
top_5_median = df[df['rank'].le(5)]['total'].median()

# Median total for all nations
all_median = df['total'].median()

print(f"Final Answer: {top_5_median}, {all_median}")