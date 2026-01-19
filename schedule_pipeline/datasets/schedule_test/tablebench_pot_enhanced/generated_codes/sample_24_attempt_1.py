import pandas as pd

df = pd.read_csv('table.csv')
# Filter data from 2000 to 2008
df_filtered = df[df['year'].astype(int) >= 2000]
df_filtered = df_filtered[df_filtered['year'].astype(int) <= 2008]

# Convert all nationality columns to numeric
nationality_columns = ['indians admitted', 'pakistanis admitted', 'sri lankans admitted', 'bangladeshis admitted', 'nepalis admitted']
df_filtered[nationality_columns] = df_filtered[nationality_columns].apply(pd.to_numeric)

# Calculate total admissions
total_admissions = df_filtered[nationality_columns].sum().sum()
print(f"Final Answer: {total_admissions}")