import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2000 to 2010, excluding the total row
df_filtered = df[df['year'].between(2000, 2010)]
# Calculate the average number of Indians admitted
average_indians = df_filtered['indians admitted'].mean()
print(f"Final Answer: {average_indians:.0f}")