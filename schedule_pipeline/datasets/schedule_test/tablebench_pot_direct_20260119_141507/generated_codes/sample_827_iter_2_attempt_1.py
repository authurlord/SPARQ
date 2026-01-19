import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' column to numeric (handles string values like '51', '21')
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Get total medals for top 5 nations (rank 1 to 5)
top_5_totals = df[df['rank'].le(5)]['total']
median_top_5 = top_5_totals.median()

# Get total medals for all nations
all_totals = df['total']
median_all = all_totals.median()

print(f"Final Answer: {median_top_5}, {median_all}")