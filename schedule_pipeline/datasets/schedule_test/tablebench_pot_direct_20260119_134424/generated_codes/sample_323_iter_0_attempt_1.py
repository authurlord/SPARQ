import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where Notes contain 'PB'
pb_row = df[df['Notes'] == '2:12:19 PB']
# Extract the Year and Venue
year_pb = pb_row['Year'].values[0]
venue_pb = pb_row['Venue'].values[0]
print(f"Final Answer: {year_pb}, {venue_pb}")