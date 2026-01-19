import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2005 to 2009 inclusive
filtered_df = df[df['year'].between(2005, 2009)]
# Calculate the average of the 'total' column
average_total = filtered_df['total'].mean()
print(f"Final Answer: {average_total:.2f}")