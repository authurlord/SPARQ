import pandas as pd

df = pd.read_csv('table.csv')

# Convert year column to string and extract numeric years
df['year'] = df['year'].astype(str)
df['year'] = df['year'].str.extract(r'(\d{4})').fillna(0).astype(int)

# Filter rows where year is between 2000 and 2007 inclusive
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]

# Calculate average quantity
average_quantity = filtered_df['quantity'].mean()

print(f"Final Answer: {average_quantity:.1f}")