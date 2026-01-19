import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'profit (us billion)' to numeric, handling negative signs correctly
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'], errors='coerce')
# Get the top 5 companies (first 5 rows)
top_5_profit = df['profit (us billion)'].head(5).mean()
print(f"Final Answer: {top_5_profit:.1f}")