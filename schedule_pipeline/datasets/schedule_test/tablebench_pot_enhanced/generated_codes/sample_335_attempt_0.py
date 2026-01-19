import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Heptathlon events at South American Championships in Lima, Peru
filtered_data = df[(df['Event'] == 'Heptathlon') & (df['Venue'] == 'Lima, Peru')]
# Extract the year and points score
year = filtered_data['Year'].values[0]
points = filtered_data['Notes'].values[0]
print(f"Final Answer: {year}, {points}")