import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '100 m' and Competition is 'African Junior Championships' and Position is '1st'
filtered_rows = df[(df['Event'] == '100 m') & (df['Competition'] == 'African Junior Championships') & (df['Position'] == '1st')]
# Get the Year of that row
year = filtered_rows.iloc[0]['Year'] if not filtered_rows.empty else None
print(f"Final Answer: {year}")