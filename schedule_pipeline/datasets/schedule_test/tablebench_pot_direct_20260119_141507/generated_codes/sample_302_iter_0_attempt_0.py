import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter matches in 2001 and where result is 'win'
df_2001_win = df[(df['year'] == '2001') & (df['result'] == 'win')]

# Group by location and count wins
win_counts = df_2001_win.groupby('location').size().reset_index(name='win_count')

# Find the city with the most wins
top_city = win_counts.loc[win_counts['win_count'].idxmax(), 'location']

# Get the first win (earliest date) in that city
first_win_in_top_city = df_2001_win[df_2001_win['location'] == top_city].sort_values(by='date').iloc[0]

# Extract the score of the first win
score_of_first_win = first_win_in_top_city['score']

print(f"Final Answer: {top_city}, {score_of_first_win}")