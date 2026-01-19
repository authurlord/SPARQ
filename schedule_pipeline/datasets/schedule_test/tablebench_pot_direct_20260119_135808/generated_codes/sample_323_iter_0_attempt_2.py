import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where Notes contain 'PB'
pb_row = df[df['Notes'].str.contains('PB', na=False)]
# Extract the Year and Venue
year_pb = pb_row['Year'].iloc[0]
venue_pb = pb_row['Venue'].iloc[0]
print(f"Final Answer: {year_pb}, {venue_pb}")