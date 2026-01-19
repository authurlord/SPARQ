import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean and convert 'sunshine (hrs / year)' to integer, removing spaces
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)

# Find the year with the most and least sunshine
max_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmax()]
min_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmin()]

# Calculate total inclement weather days (snow + storms + fog) for both years
max_inclement_days = max_sunshine_year['snow (days / year)'] + max_sunshine_year['storms (days / year)'] + max_sunshine_year['fog (days / year)']
min_inclement_days = min_sunshine_year['snow (days / year)'] + min_sunshine_year['storms (days / year)'] + min_sunshine_year['fog (days / year)']

# Compute the difference
difference = abs(max_inclement_days - min_inclement_days)

print(f"Final Answer: {difference}")