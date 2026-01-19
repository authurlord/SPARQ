import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Notes' contains 'PB'
pb_row = df[df['Notes'].str.contains('PB', case=False, na=False)]
# If found, extract Year and Venue
if not pb_row.empty:
    year = pb_row.iloc[0]['Year']
    venue = pb_row.iloc[0]['Venue']
    print(f"Final Answer: {year}, {venue}")
else:
    print("Final Answer: None")