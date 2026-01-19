import pandas as pd

df = pd.read_csv('table.csv')
# Convert profit column to numeric, handling negative signs properly
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'], errors='coerce')
# Get top 5 companies by rank
top_5_profit = df['profit (us billion)'].head(5).mean()
print(f"Final Answer: {top_5_profit:.1f}")