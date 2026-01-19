import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'total votes' and '% of popular vote' to numeric
df['total votes'] = pd.to_numeric(df['total votes'])
df['% of popular vote'] = pd.to_numeric(df['% of popular vote'].str.replace('%', ''))

# Extract years and values
years = df['election'].astype(int)
votes = df['total votes']
vote_percentage = df['% of popular vote']

# Use last two data points to calculate trend
last_year = years.iloc[-1]
last_vote = votes.iloc[-1]
last_percent = vote_percentage.iloc[-1]

# Calculate average annual increase from 2009 to 2013
year_diff = last_year - years.iloc[-2]
vote_increase = (last_vote - votes.iloc[-2]) / year_diff
percent_increase = (last_percent - vote_percentage.iloc[-2]) / year_diff

# Predict for next election (2017)
predicted_votes = last_vote + vote_increase * 4
predicted_percent = last_percent + percent_increase * 4

print(f"Final Answer: {predicted_votes:.0f}, {predicted_percent:.2f}%")