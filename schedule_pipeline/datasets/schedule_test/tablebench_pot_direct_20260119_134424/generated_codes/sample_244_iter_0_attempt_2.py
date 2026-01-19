import pandas as pd

df = pd.read_csv('table.csv')

# Clean the sunshine column by removing spaces and converting to integer
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)

# Calculate total inclement weather days (snow + storms + fog)
df['inclement_days'] = df['snow (days / year)'].astype(int) + df['storms (days / year)'].astype(int) + df['fog (days / year)'].astype(int)

# Find the year with the most sunshine and least sunshine
max_sunshine_idx = df['sunshine (hrs / year)'].idxmax()
min_sunshine_idx = df['sunshine (hrs / year)'].idxmin()

# Get the inclement weather days for those years
max_sunshine_inclement = df.loc[max_sunshine_idx, 'inclement_days']
min_sunshine_inclement = df.loc[min_sunshine_idx, 'inclement_days']

# Calculate the difference
difference = abs(max_sunshine_inclement - min_sunshine_inclement)

print(f"Final Answer: {difference}")