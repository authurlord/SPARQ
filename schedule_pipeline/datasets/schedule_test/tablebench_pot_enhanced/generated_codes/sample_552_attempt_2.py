import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2000 to 2010
df_filtered = df[df['year'].between('2000', '2010')]
# Convert 'indians admitted' to integer and calculate mean
mean_indians = df_filtered['indians admitted'].astype(int).mean()
print(f"Final Answer: {mean_indians:.1f}")