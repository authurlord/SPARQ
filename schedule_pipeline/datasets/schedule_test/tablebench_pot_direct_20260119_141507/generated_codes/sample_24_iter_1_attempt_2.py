import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer for proper filtering
df['year'] = df['year'].astype(int)
# Filter years from 2000 to 2008 inclusive
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2008)]
# Sum all admitted values across all nationalities
total_admitted = filtered_df[['indians admitted', 'pakistanis admitted', 'sri lankans admitted', 
                              'bangladeshis admitted', 'nepalis admitted']].sum().sum()
print(f"Final Answer: {total_admitted}")