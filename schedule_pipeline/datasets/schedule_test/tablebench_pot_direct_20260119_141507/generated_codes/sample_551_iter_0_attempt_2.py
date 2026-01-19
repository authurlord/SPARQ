import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'total' row
df = df[df['year'] != 'total']

# Convert 'wins' to integer for calculation
df['wins'] = df['wins'].astype(int)

# Calculate the year-over-year increase in wins
df['win_increase'] = df['wins'].diff()

# Find the year with the largest increase (excluding the first year which has no prior)
max_increase_row = df[df['win_increase'] == df['win_increase'].max()]
year_with_max_increase = max_increase_row.iloc[0]['year']

print(f"Final Answer: {year_with_max_increase}")