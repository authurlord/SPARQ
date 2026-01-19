import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer for proper filtering
df['year'] = pd.to_numeric(df['year'], errors='coerce')
# Filter rows where year is between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]
# Calculate the average of 'quantity' column
average_quantity = filtered_df['quantity'].mean()
print(f"Final Answer: {average_quantity:.1f}")