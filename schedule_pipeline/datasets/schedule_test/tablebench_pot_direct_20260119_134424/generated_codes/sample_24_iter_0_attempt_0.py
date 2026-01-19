import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2000 to 2008
df_filtered = df[df['year'].astype(int) >= 2000][df['year'].astype(int) <= 2008]
# Convert admission columns to numeric and sum them
admission_columns = ['indians admitted', 'pakistanis admitted', 'sri lankans admitted', 'bangladeshis admitted', 'nepalis admitted']
total_admissions = df_filtered[admission_columns].sum().sum()
print(f"Final Answer: {total_admissions}")