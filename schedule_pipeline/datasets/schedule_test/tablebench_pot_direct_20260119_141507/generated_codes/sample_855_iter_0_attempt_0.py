import pandas as pd

df = pd.read_csv('table.csv')

# Extract the relevant columns
col_2005_06 = '2005 - 06'
col_2008_09 = '2008 - 09'

# Calculate the difference for each station
df['diff'] = abs(df[col_2005_06].astype(int) - df[col_2008_09].astype(int))

# Sort by difference and get the top 2 stations with smallest difference
smallest_diff_stations = df.sort_values(by='diff', ascending=True).head(2)['station name'].tolist()

print(f"Final Answer: {smallest_diff_stations[0]}, {smallest_diff_stations[1]}")