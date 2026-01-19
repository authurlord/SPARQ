import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 2002 and 2005 inclusive
filtered_df = df[(df['year'] >= 2002) & (df['year'] <= 2005)]
# Convert issue price to numeric and compute average
average_price = filtered_df['issue price'].astype(float).mean()
print(f"Final Answer: {average_price:.2f}")