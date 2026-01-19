import pandas as pd

df = pd.read_csv('table.csv')
# Select top 5 companies (first 5 rows)
top_5 = df.head(5)
# Extract profit column and convert to float (handling strings like '-16')
profits = pd.to_numeric(top_5['profit (us billion)'], errors='coerce')
# Calculate mean of non-null profits
avg_profit = profits.mean()
print(f"Final Answer: {avg_profit:.1f}")