import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'year' column by converting to string and removing any commas
df['year'] = df['year'].astype(str).str.replace(',', '')

# Filter rows where year is between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'].astype(int) >= 2000) & (df['year'].astype(int) <= 2007)]

# Calculate the average quantity for the filtered rows
average_quantity = filtered_df['quantity'].mean()

print(f"Final Answer: {average_quantity:.1f}")