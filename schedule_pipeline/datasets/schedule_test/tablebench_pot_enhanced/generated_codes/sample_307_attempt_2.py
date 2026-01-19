import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the specific conditions: Year 2008, Competition 'Asian Junior Championships', Position '2nd'
filtered_data = df[(df['Year'] == '2008') & (df['Competition'] == 'Asian Junior Championships') & (df['Position'] == '2nd')]
# Extract the event
event = filtered_data['Event'].values[0]
print(f"Final Answer: {event}")