import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' column to numeric, coercing errors to NaN (though all values should be valid)
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Top 5 rankings (rank 1 to 5)
top_5_total = df[df['rank'].isin(range(1, 6))]['total']
median_top_5 = top_5_total.median()

# All countries
median_all = df['total'].median()

print(f"Final Answer: {median_top_5}, {median_all}")