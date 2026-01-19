import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer for proper sorting
df['year'] = pd.to_numeric(df['year'])
# Calculate the difference in tropical storms from the previous year
df['increase'] = df['number of tropical storms'].diff()
# Find the year with the maximum increase (excluding the first year)
max_increase_year = df.loc[df['increase'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")