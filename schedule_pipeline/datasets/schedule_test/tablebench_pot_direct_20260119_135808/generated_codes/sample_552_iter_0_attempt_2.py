import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2000 to 2010
df_filtered = df[df['year'].between('2000', '2010')]
# Convert 'indians admitted' to integer
df_filtered['indians admitted'] = pd.to_numeric(df_filtered['indians admitted'])
# Calculate average
average_indians = df_filtered['indians admitted'].mean()
print(f"Final Answer: {average_indians:.1f}")