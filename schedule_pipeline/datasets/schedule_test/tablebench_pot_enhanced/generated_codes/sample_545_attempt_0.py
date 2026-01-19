import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer and filter data for years 2005 to 2009
filtered_df = df[(df['year'] >= 2005) & (df['year'] <= 2009)]
# Calculate average total
average_total = filtered_df['total'].mean()
print(f"Final Answer: {average_total:.2f}")