import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('table.csv')

# Convert original air date to date objects
df['original air date'] = pd.to_datetime(df['original air date'], format='%B %d, %Y', errors='coerce')

# Filter episodes directed by Kyle Dunlevy between Sep 2012 and Feb 2013
kyle_episodes = df[(df['directed by'] == 'kyle dunlevy') & 
                   (df['original air date'] >= '2012-09-01') & 
                   (df['original air date'] <= '2013-02-29')]

# Sort by date
kyle_episodes = kyle_episodes.sort_values('original air date')

# Extract date and viewership
dates = kyle_episodes['original air date']
viewership = kyle_episodes['us viewers (million)']

# Create time index (in days since start)
time_days = (dates - dates.iloc[0]).dt.days

# Linear regression: viewership ~ time_days
slope, intercept = np.polyfit(time_days, viewership, 1)

# Project to March 2013 (assume March 1, 2013)
target_date = '2013-03-01'
target_day = (pd.to_datetime(target_date) - dates.iloc[0]).days
projected_viewership = slope * target_day + intercept

print(f"Final Answer: {projected_viewership:.2f}")