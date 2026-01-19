import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer
df['year'] = df['year'].astype(int)
# Filter data from 2000 to 2008 inclusive
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2008)]
# Sum all admitted numbers across all nationalities
total_admitted = filtered_df[['indians admitted', 'pakistanis admitted', 'sri lankans admitted', 
                              'bangladeshis admitted', 'nepalis admitted']].sum().sum()
print(f"Final Answer: {total_admitted}")