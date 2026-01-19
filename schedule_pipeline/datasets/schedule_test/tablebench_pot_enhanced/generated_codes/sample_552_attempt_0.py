import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2000 to 2010
df_filtered = df[df['year'].between('2000', '2010')]
# Convert 'indians admitted' to numeric and calculate average
avg_indians = df_filtered['indians admitted'].astype(int).mean()
print(f"Final Answer: {avg_indians:.1f}")