import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'wins' to integer
df['wins'] = pd.to_numeric(df['wins'])
# Calculate the difference in wins from the previous year
df['win_increase'] = df['wins'].diff()
# Find the year with the maximum increase (excluding the first year)
max_increase_year = df.loc[df['win_increase'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")