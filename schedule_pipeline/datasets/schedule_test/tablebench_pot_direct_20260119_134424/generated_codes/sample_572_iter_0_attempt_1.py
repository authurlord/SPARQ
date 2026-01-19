import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' column to integer
df['typhus'] = pd.to_numeric(df['typhus'])
# Calculate the difference in typhus cases from the previous year
df['diff'] = df['typhus'].diff()
# Find the year with the most negative difference (largest decrease)
min_diff_year = df.loc[df['diff'].idxmin(), 'year']
print(f"Final Answer: {min_diff_year}")