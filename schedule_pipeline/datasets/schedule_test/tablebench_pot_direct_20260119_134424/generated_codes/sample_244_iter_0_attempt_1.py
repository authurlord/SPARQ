import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'sunshine (hrs / year)' to numeric (remove spaces and handle formatting)
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)

# Convert other columns to numeric
df['rain (mm / year)'] = df['rain (mm / year)'].str.replace(' ', '').astype(int)
df['snow (days / year)'] = df['snow (days / year)'].str.replace(' ', '').astype(int)
df['storms (days / year)'] = df['storms (days / year)'].str.replace(' ', '').astype(int)
df['fog (days / year)'] = df['fog (days / year)'].str.replace(' ', '').astype(int)

# Calculate total inclement weather days (snow + storms + fog)
df['inclement_days'] = df['snow (days / year)'] + df['storms (days / year)'] + df['fog (days / year)']

# Find the year with the most and least sunshine
max_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmax()]
min_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmin()]

# Get the inclement weather days for both years
max_sunshine_inclement = max_sunshine_year['inclement_days']
min_sunshine_inclement = min_sunshine_year['inclement_days']

# Calculate the difference
difference = abs(max_sunshine_inclement - min_sunshine_inclement)

print(f"Final Answer: {difference}")