import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer for proper comparison
df['year'] = df['year'].astype(str).str.extract('(\d+)').astype(int)
# Filter rows where year is between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]
# Calculate the average quantity
avg_quantity = filtered_df['quantity'].mean()
print(f"Final Answer: {avg_quantity:.1f}")