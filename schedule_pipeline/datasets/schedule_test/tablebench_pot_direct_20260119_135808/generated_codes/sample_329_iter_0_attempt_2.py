import pandas as pd

df = pd.read_csv('table.csv')
# Filter for year 2005 and position '4th'
filtered_data = df[(df['Year'] == '2005') & (df['Position'] == '4th')]
# Extract event and venue
event = filtered_data['Event'].iloc[0]
venue = filtered_data['Venue'].iloc[0]
print(f"Final Answer: {event}, {venue}")