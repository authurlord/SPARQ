import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer for filtering
df['year'] = df['year'].astype(int)
# Filter rows for years between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]
# Calculate average quantity
avg_quantity = filtered_df['quantity'].mean()
print(f"Final Answer: {avg_quantity:.1f}")