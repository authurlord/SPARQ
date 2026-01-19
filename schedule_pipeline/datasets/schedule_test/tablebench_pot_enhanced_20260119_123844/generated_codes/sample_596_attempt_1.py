import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'profit (us billion)' to numeric, handling negative values
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'], errors='coerce')
# Calculate average profit for top 5 companies
avg_profit_top5 = df['profit (us billion)'].head(5).mean()
print(f"Final Answer: {avg_profit_top5:.1f}")