import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: remove spaces in numeric strings
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)
df['rain (mm / year)'] = df['rain (mm / year)'].str.replace(' ', '').astype(int)
df['snow (days / year)'] = df['snow (days / year)'].str.replace(' ', '').astype(int)
df['storms (days / year)'] = df['storms (days / year)'].str.replace(' ', '').astype(int)
df['fog (days / year)'] = df['fog (days / year)'].str.replace(' ', '').astype(int)

# Compute correlation between snow and storms
corr_snow_storms = df['snow (days / year)'].corr(df['storms (days / year)'])

# Compute correlation between snow and fog
corr_snow_fog = df['snow (days / year)'].corr(df['fog (days / year)'])

# Determine which has greater impact based on absolute correlation
if abs(corr_snow_storms) > abs(corr_snow_fog):
    impact = 'storms'
else:
    impact = 'fog'

print(f"Final Answer: {impact}")