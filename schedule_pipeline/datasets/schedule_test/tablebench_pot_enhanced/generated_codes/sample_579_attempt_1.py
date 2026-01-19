import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'year' to integer and filter for years between 2000 and 2007
df['year'] = df['year'].astype(str).str.extract('(\d{4})').astype(int)
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]

# Calculate average quantity
average_quantity = filtered_df['quantity'].mean()

print(f"Final Answer: {average_quantity:.1f}")