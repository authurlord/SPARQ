import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' column to numeric to ensure proper calculation
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Get the total medals for top 5 rankings (rank 1 to 5)
top_5_total = df[df['rank'].astype(int).le(5)]['total']

# Compute median for top 5 and all nations
median_top_5 = top_5_total.median()
median_all = df['total'].median()

print(f"Final Answer: {median_top_5}, {median_all}")