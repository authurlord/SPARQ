import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the specific conditions
filtered_df = df[(df['Competition'] == 'South American Championships') &
                 (df['Venue'] == 'Lima, Peru') &
                 (df['Event'] == 'Heptathlon') &
                 (df['Position'] == '1st')]

# Extract year and points score
year = filtered_df['Year'].iloc[0]
points = filtered_df['Notes'].iloc[0]

print(f"Final Answer: {year}, {points}")