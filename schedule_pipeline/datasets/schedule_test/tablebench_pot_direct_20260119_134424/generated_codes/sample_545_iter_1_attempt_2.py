import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer to enable numerical comparison
df['year'] = pd.to_numeric(df['year'], errors='coerce')
# Filter rows where year is between 2005 and 2009 (inclusive)
filtered_df = df[(df['year'] >= 2005) & (df['year'] <= 2009)]
# Calculate the average of 'total' column for the filtered data
average_total = filtered_df['total'].mean()
print(f"Final Answer: {average_total:.2f}")