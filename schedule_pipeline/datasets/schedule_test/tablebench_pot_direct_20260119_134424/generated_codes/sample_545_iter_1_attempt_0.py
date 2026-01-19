import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' and 'total' columns to appropriate types
df['year'] = pd.to_numeric(df['year'])
df['total'] = pd.to_numeric(df['total'])

# Filter data for years between 2005 and 2009 (inclusive)
filtered_df = df[(df['year'] >= 2005) & (df['year'] <= 2009)]

# Calculate the average total value
average_total = filtered_df['total'].mean()

print(f"Final Answer: {average_total:.2f}")