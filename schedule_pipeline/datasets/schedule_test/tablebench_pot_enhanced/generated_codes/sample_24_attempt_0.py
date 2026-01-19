import pandas as pd

df = pd.read_csv('table.csv')
# Filter data from 2000 to 2008
df_filtered = df[df['year'].astype(int) >= 2000]
df_filtered = df_filtered[df_filtered['year'].astype(int) <= 2008]

# Sum all admission columns (excluding 'year')
total_admissions = df_filtered[['indians admitted', 'pakistanis admitted', 'sri lankans admitted', 'bangladeshis admitted', 'nepalis admitted']].sum().sum()

print(f"Final Answer: {total_admissions}")