import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'election' to integer for proper numerical analysis
df['election'] = pd.to_numeric(df['election'])

# Extract the relevant columns
years = df['election'].values
total_votes = df['total votes'].values
popular_vote_percent = df['% of popular vote'].str.replace('%', '').astype(float).values

# Perform linear regression for total votes
slope_votes, intercept_votes = np.polyfit(years, total_votes, 1)
predicted_total_votes = slope_votes * 2017 + intercept_votes

# Perform linear regression for percentage of popular vote
slope_percent, intercept_percent = np.polyfit(years, popular_vote_percent, 1)
predicted_percent = slope_percent * 2017 + intercept_percent

print(f"Final Answer: {predicted_total_votes:.0f}, {predicted_percent:.2f}%")