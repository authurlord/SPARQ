import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' column to integer
df['typhus'] = pd.to_numeric(df['typhus'])
# Calculate the difference between consecutive years
df['diff'] = df['typhus'].diff()
# Find the year with the maximum decrease (most negative difference)
min_decrease_year = df.loc[df['diff'].idxmin(), 'year']
print(f"Final Answer: {min_decrease_year}")