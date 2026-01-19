import pandas as pd

df = pd.read_csv('table.csv')

# Extract the relevant columns
col_2005_06 = '2005 - 06'
col_2008_09 = '2008 - 09'

# Convert the relevant columns to numeric (in case of string formatting)
df_numeric = df.astype({col_2005_06: 'int', col_2008_09: 'int'})

# Calculate the absolute difference for each station
df['diff'] = abs(df[col_2005_06].astype(int) - df[col_2008_09].astype(int))

# Sort by difference and get the two stations with smallest difference
sorted_stations = df.sort_values(by='diff').head(2)

# Extract station names
smallest_diff_stations = sorted_stations['station name'].tolist()

print(f"Final Answer: {smallest_diff_stations[0]}, {smallest_diff_stations[1]}")