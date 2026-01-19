import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the specific competition and year
filtered_df = df[(df['Year'] == '2008') & (df['Competition'] == 'Asian Junior Championships')]
# Extract the event
event = filtered_df['Event'].values[0]
print(f"Final Answer: {event}")