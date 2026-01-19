import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'quantity' to numeric, coercing errors to NaN
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

# Filter rows within the year range 2000 to 2007
filtered_df = df[(df['year'].astype(int) >= 2000) & (df['year'].astype(int) <= 2007)]

# Calculate the average quantity
average_quantity = filtered_df['quantity'].mean()

print(f"Final Answer: {average_quantity:.1f}")