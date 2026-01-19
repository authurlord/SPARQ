import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2000 to 2008
filtered_df = df[df['year'].between(2000, 2008)]
# Sum all admitted numbers across nationalities
total_admitted = filtered_df[['indians admitted', 'pakistanis admitted', 'sri lankans admitted', 
                              'bangladeshis admitted', 'nepalis admitted']].sum().sum()
print(f"Final Answer: {total_admitted}")