import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year is 2005 and Position is '4th'
filtered_row = df[(df['Year'] == '2005') & (df['Position'] == '4th')]
# Extract Event and Venue
event = filtered_row['Event'].values[0]
venue = filtered_row['Venue'].values[0]
print(f"Final Answer: {event}, {venue}")