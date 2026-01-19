import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year is 2005 and Position is '4th'
filtered_df = df[(df['Year'] == '2005') & (df['Position'] == '4th')]
# Extract Event and Venue
event = filtered_df['Event'].iloc[0]
venue = filtered_df['Venue'].iloc[0]
print(f"Final Answer: {event}, {venue}")