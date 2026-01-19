import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to string for safe comparison
df['year'] = df['year'].astype(str)
# Filter rows where year is between 2000 and 2007
filtered_df = df[df['year'].str.contains('200[0-7]', na=False)]
# Convert 'quantity' to numeric, coercing errors to NaN
filtered_df['quantity'] = pd.to_numeric(filtered_df['quantity'], errors='coerce')
# Calculate average quantity
average_quantity = filtered_df['quantity'].mean()
print(f"Final Answer: {average_quantity:.1f}")