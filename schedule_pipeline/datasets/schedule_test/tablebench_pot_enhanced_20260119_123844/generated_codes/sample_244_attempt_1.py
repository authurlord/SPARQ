import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert sunshine column to integer
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)

# Calculate total inclement weather days
df['inclement_weather_days'] = df['snow (days / year)'].astype(int) + df['storms (days / year)'].astype(int) + df['fog (days / year)'].astype(int)

# Find the year with most and least sunshine
max_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmax()]
min_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmin()]

# Get inclement weather days for both years
max_sunshine_inclement = max_sunshine_year['inclement_weather_days']
min_sunshine_inclement = min_sunshine_year['inclement_weather_days']

# Calculate difference
difference = abs(max_sunshine_inclement - min_sunshine_inclement)

print(f"Final Answer: {difference}")