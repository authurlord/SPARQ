import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer, handling cases like "2010 , 2012" by taking the first year
df['year'] = df['year'].astype(str).str.split(',').str[0].astype(int)
# Filter rows where year is between 2000 and 2007 (inclusive)
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2007)]
# Calculate average quantity
avg_quantity = filtered_df['quantity'].mean()
print(f"Final Answer: {avg_quantity:.1f}")