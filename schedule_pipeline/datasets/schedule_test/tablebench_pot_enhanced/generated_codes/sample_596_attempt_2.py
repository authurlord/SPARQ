import pandas as pd

df = pd.read_csv('table.csv')
# Convert profit column to numeric, handling negative signs properly
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'], errors='coerce')
# Get the top 5 companies and calculate average profit
top_5_profit_avg = df['profit (us billion)'].head(5).mean()
print(f"Final Answer: {top_5_profit_avg:.1f}")