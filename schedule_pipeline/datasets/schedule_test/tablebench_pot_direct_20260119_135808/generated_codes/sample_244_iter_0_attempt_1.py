import pandas as pd

df = pd.read_csv('table.csv')

# Clean the columns by removing spaces and converting to integers
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)
df['rain (mm / year)'] = df['rain (mm / year)'].str.replace(' ', '').astype(int)
df['snow (days / year)'] = df['snow (days / year)'].str.replace(' ', '').astype(int)
df['storms (days / year)'] = df['storms (days / year)'].str.replace(' ', '').astype(int)
df['fog (days / year)'] = df['fog (days / year)'].str.replace(' ', '').astype(int)

# Calculate total inclement weather days
df['inclement days'] = df['snow (days / year)'] + df['storms (days / year)'] + df['fog (days / year)']

# Find the year with the most and least sunshine
max_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmax()]
min_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmin()]

# Get the inclement days for both years
max_sunshine_inclement = max_sunshine_year['inclement days']
min_sunshine_inclement = min_sunshine_year['inclement days']

# Compute the difference
difference = abs(max_sunshine_inclement - min_sunshine_inclement)

print(f"Final Answer: {difference}")