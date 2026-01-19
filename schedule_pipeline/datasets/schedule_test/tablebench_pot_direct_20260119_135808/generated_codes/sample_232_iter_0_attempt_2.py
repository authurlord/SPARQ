import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: remove spaces in numeric strings and convert to numeric
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)
df['rain (mm / year)'] = df['rain (mm / year)'].str.replace(' ', '').astype(int)
df['snow (days / year)'] = df['snow (days / year)'].str.replace(' ', '').astype(int)
df['storms (days / year)'] = df['storms (days / year)'].str.replace(' ', '').astype(int)
df['fog (days / year)'] = df['fog (days / year)'].str.replace(' ', '').astype(int)

# Calculate correlation between snow and storms, and snow and fog
corr_snow_storms = df['snow (days / year)'].corr(df['storms (days / year)'])
corr_snow_fog = df['snow (days / year)'].corr(df['fog (days / year)'])

# Determine which has greater impact (higher absolute correlation)
if abs(corr_snow_storms) > abs(corr_snow_fog):
    impact = 'storms (days / year)'
else:
    impact = 'fog (days / year)'

print(f"Final Answer: {impact}")