import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'sunshine (hrs / year)' column by removing spaces and converting to numeric
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(float)

# Calculate total inclement weather days per year (snow + storms + fog)
df['inclement_weather'] = df['snow (days / year)'] + df['storms (days / year)'] + df['fog (days / year)']

# Find the year with the most and least sunshine
max_sunshine_row = df.loc[df['sunshine (hrs / year)'].idxmax()]
min_sunshine_row = df.loc[df['sunshine (hrs / year)'].idxmin()]

# Get inclement weather days for those years
max_inclement = max_sunshine_row['inclement_weather']
min_inclement = min_sunshine_row['inclement_weather']

# Compute the difference
difference = max_inclement - min_inclement

print(f"Final Answer: {difference}")