import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer and filter data from 2005 to 2009
filtered_df = df[(df['year'] >= 2005) & (df['year'] <= 2009)]
# Calculate average total
avg_total = filtered_df['total'].mean()
print(f"Final Answer: {avg_total:.2f}")