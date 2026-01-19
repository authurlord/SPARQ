import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years between 2002 and 2005 inclusive
filtered_df = df[(df['year'].between(2002, 2005))]
# Convert issue price to float and compute mean
mean_issue_price = filtered_df['issue price'].str.replace('$', '').astype(float).mean()
print(f"Final Answer: {mean_issue_price:.2f}")