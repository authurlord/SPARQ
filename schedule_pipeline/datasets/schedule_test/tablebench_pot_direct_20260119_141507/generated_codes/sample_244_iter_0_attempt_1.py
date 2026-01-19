import pandas as pd

df = pd.read_csv('table.csv')

# Convert all values to numeric (handling potential formatting issues)
df['sunshine (hrs / year)'] = pd.to_numeric(df['sunshine (hrs / year)'].str.replace(' ', ''), errors='coerce')
df['rain (mm / year)'] = pd.to_numeric(df['rain (mm / year)'], errors='coerce')
df['snow (days / year)'] = pd.to_numeric(df['snow (days / year)'], errors='coerce')
df['storms (days / year)'] = pd.to_numeric(df['storms (days / year)'], errors='coerce')
df['fog (days / year)'] = pd.to_numeric(df['fog (days / year)'], errors='coerce')

# Calculate total inclement weather days per year
df['inclement_days'] = df['snow (days / year)'] + df['storms (days / year)'] + df['fog (days / year)']

# Find the year with the most and least sunshine
max_sunshine_row = df.loc[df['sunshine (hrs / year)'].idxmax()]
min_sunshine_row = df.loc[df['sunshine (hrs / year)'].idxmin()]

# Get inclement days for those years
max_inclement = max_sunshine_row['inclement_days']
min_inclement = min_sunshine_row['inclement_days']

# Compute the difference
difference = max_inclement - min_inclement

print(f"Final Answer: {difference}")