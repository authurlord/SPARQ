import pandas as pd

df = pd.read_csv('table.csv')
# Filter for rows where Notes contain 'PB' and Competition is 'Irish National Marathon Championship'
pb_row = df[(df['Notes'] == '2:12:19 PB') & (df['Competition'] == 'Irish National Marathon Championship')]
# Extract Year and Venue
year = pb_row['Year'].values[0]
venue = pb_row['Venue'].values[0]
print(f"Final Answer: {year}, {venue}")