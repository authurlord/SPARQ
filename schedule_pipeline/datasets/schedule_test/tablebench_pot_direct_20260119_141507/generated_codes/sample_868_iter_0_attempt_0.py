import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the difference between 'property taxes' and 'investment earnings'
df['difference'] = abs(df['property taxes'] - df['investment earnings'])
# Find the year with the maximum difference
max_diff_year = df.loc[df['difference'].idxmax(), 'year']
print(f"Final Answer: {max_diff_year}")