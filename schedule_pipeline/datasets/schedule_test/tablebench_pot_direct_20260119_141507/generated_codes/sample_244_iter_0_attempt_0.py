import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the sunshine column by removing spaces and converting to integer
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)

# Calculate inclement weather days (snow + storms + fog)
df['inclement_weather'] = df['snow (days / year)'] + df['storms (days / year)'] + df['fog (days / year)']

# Find the year with max and min sunshine
max_sunshine_row = df.loc[df['sunshine (hrs / year)'].idxmax()]
min_sunshine_row = df.loc[df['sunshine (hrs / year)'].idxmin()]

# Get inclement weather days for those years
max_inclement = max_sunshine_row['inclement_weather']
min_inclement = min_sunshine_row['inclement_weather']

# Calculate the difference
difference = max_inclement - min_inclement

print(f"Final Answer: {difference}")