import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 2005 and 2009 inclusive
filtered_df = df[(df['year'].astype(int) >= 2005) & (df['year'].astype(int) <= 2009)]
# Calculate the mean of the 'total' column after converting to float
average_total = filtered_df['total'].str.replace(',', '').astype(float).mean()
print(f"Final Answer: {average_total:.2f}")