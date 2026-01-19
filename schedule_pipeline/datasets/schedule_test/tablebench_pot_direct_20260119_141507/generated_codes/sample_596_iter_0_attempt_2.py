import pandas as pd

df = pd.read_csv('table.csv')
# Extract the first 5 rows (top 5 companies)
top_5 = df.head(5)
# Clean the 'profit (us billion)' column by removing spaces and converting to float
top_5['profit (us billion)'] = top_5['profit (us billion)'].str.replace(' - ', '-').astype(float)
# Calculate the average profit
average_profit = top_5['profit (us billion)'].mean()
print(f"Final Answer: {average_profit:.1f}")