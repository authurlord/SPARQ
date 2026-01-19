import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where Year is 2008, Competition is 'Asian Junior Championships', and Position is '2nd'
filtered_row = df[(df['Year'] == '2008') & (df['Competition'] == 'Asian Junior Championships') & (df['Position'] == '2nd')]
# Extract the 'Event' value
event = filtered_row['Event'].values[0]
print(f"Final Answer: {event}")