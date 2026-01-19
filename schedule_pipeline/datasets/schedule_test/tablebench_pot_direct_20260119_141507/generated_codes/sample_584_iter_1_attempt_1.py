import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer for proper numerical comparison
df['year'] = df['year'].astype(int)
# Filter rows where year is between 2002 and 2005 (inclusive)
filtered_df = df[(df['year'] >= 2002) & (df['year'] <= 2005)]
# Calculate the average issue price
avg_issue_price = filtered_df['issue price'].mean()
print(f"Final Answer: {avg_issue_price:.2f}")