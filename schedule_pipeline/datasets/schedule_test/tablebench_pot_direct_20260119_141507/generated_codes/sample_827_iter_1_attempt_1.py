import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rank' to integer for proper filtering
df['rank'] = df['rank'].astype(int)

# Median total for top 5 nations (rank <= 5)
top_5_total = df[df['rank'] <= 5]['total'].median()

# Median total for all nations
all_total_median = df['total'].median()

print(f"Final Answer: {top_5_total}, {all_total_median}")