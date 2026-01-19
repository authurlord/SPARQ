import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'election' to integer for proper indexing
df['election'] = df['election'].astype(int)

# Extract the relevant columns and convert to numeric
votes = df['total votes'].astype(int)
percent_vote = df['% of popular vote'].str.replace('%', '').astype(float)

# Create a time variable (election year)
years = df['election'].values

# Fit a linear model for total votes and percentage of popular vote
# Using last 4 data points for better trend estimation
last_years = years[-4:]
last_votes = votes[-4:]
last_percent = percent_vote[-4:]

# Linear regression for total votes
slope_votes, intercept_votes = np.polyfit(last_years, last_votes, 1)
predicted_votes = slope_votes * (years[-1] + 4) + intercept_votes

# Linear regression for percentage of popular vote
slope_percent, intercept_percent = np.polyfit(last_years, last_percent, 1)
predicted_percent = slope_percent * (years[-1] + 4) + intercept_percent

# Round to nearest whole number and one decimal place respectively
final_votes = round(predicted_votes)
final_percent = round(predicted_percent, 2)

print(f"Final Answer: {final_votes}, {final_percent}%")