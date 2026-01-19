import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'quantity' to numeric, coercing errors to NaN to handle invalid entries
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

# Filter rows where year is between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'].astype(int) >= 2000) & (df['year'].astype(int) <= 2007)]

# Calculate the average quantity
average_quantity = filtered_df['quantity'].mean()

print(f"Final Answer: {average_quantity:.1f}")