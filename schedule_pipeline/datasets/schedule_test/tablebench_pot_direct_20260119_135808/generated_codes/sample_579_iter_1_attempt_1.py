import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'year' column: extract the first year if multiple years are listed
df['year'] = df['year'].astype(str).str.split(',').str[0]

# Convert 'year' to integer
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Filter rows where year is between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]

# Calculate average quantity
avg_quantity = filtered_df['quantity'].mean()

print(f"Final Answer: {avg_quantity:.1f}")