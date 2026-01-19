import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Heptathlon, South American Championships, Lima, Peru, and 1st position
filtered_df = df[(df['Event'] == 'Heptathlon') & 
                 (df['Competition'] == 'South American Championships') & 
                 (df['Venue'] == 'Lima, Peru') & 
                 (df['Position'] == '1st')]

# Extract year and points score
year = filtered_df['Year'].iloc[0]
points = filtered_df['Notes'].iloc[0]

print(f"Final Answer: {year}, {points}")