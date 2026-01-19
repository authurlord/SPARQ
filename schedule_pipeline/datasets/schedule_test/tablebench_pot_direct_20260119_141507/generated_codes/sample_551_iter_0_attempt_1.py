import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'total' row
df_filtered = df[df['year'] != 'total']

# Convert 'wins' to integer for numerical computation
df_filtered['wins'] = df_filtered['wins'].astype(int)

# Calculate the year-over-year increase in wins
df_filtered['win_increase'] = df_filtered['wins'].diff()

# Find the year with the largest increase (excluding the first year which has no prior)
max_increase_row = df_filtered[df_filtered['win_increase'] == df_filtered['win_increase'].max()]

# Get the year with the largest increase
year_with_max_increase = max_increase_row.iloc[0]['year']

print(f"Final Answer: {year_with_max_increase}")