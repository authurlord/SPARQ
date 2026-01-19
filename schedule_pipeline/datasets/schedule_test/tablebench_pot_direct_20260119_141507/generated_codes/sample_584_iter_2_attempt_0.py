import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years between 2002 and 2005 inclusive
filtered_df = df[(df['year'] >= 2002) & (df['year'] <= 2005)]
# Convert 'issue price' to numeric, handling any parsing issues
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')
# Calculate the average issue price of the filtered rows
average_issue_price = filtered_df['issue price'].mean()
print(f"Final Answer: {average_issue_price:.2f}")