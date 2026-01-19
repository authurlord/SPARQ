import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Irish National Marathon Championship with PB in Notes
pb_row = df[(df['Competition'] == 'Irish National Marathon Championship') & (df['Notes'].str.contains('PB', na=False))]
# Extract year and venue
year = pb_row['Year'].values[0]
venue = pb_row['Venue'].values[0]
print(f"Final Answer: {year}, {venue}")