import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year is 2005 and Position is '4th'
filtered_rows = df[(df['Year'] == '2005') & (df['Position'] == '4th')]
# Extract the Event and Venue
event = filtered_rows['Event'].iloc[0]
venue = filtered_rows['Venue'].iloc[0]
print(f"Final Answer: {event}, {venue}")