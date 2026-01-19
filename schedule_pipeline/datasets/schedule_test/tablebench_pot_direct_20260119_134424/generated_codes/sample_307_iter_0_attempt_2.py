import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the specific conditions
filtered_df = df[(df['Year'] == '2008') & (df['Competition'] == 'Asian Junior Championships') & (df['Position'] == '2nd')]
# Extract the event
event = filtered_df['Event'].iloc[0]
print(f"Final Answer: {event}")