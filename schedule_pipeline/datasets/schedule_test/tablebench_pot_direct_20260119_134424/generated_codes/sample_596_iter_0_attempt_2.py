import pandas as pd

df = pd.read_csv('table.csv')
# Extract top 5 companies
top_5 = df.head(5)
# Convert profit column to numeric, handling negative signs and strings
top_5['profit (us billion)'] = pd.to_numeric(top_5['profit (us billion)'], errors='coerce')
# Calculate average profit
avg_profit = top_5['profit (us billion)'].mean()
print(f"Final Answer: {avg_profit:.1f}")