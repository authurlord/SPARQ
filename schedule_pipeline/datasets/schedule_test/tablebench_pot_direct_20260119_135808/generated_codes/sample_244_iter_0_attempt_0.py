import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'sunshine (hrs / year)' to numeric, handling spaces in numbers
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)

# Calculate total inclement weather days (snow + storms + fog)
df['inclement_days'] = df['snow (days / year)'].astype(int) + df['storms (days / year)'].astype(int) + df['fog (days / year)'].astype(int)

# Find the year with most and least sunshine
max_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmax()]
min_sunshine_year = df.loc[df['sunshine (hrs / year)'].idxmin()]

# Get the inclement days for both years
max_sunshine_inclement = max_sunshine_year['inclement_days']
min_sunshine_inclement = min_sunshine_year['inclement_days']

# Calculate the difference
difference = abs(max_sunshine_inclement - min_sunshine_inclement)
print(f"Final Answer: {difference}")