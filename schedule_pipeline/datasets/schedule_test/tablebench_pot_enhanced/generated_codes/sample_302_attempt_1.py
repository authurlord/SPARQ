import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2001
df_2001 = df[df['year'] == '2001']

# Group by location and count wins
wins_by_location = df_2001[df_2001['result'] == 'win'].groupby('location').size()

# Find the city with the most wins
most_wins_city = wins_by_location.idxmax()

# Get the first win in that city (sort by date)
first_win = df_2001[(df_2001['result'] == 'win') & (df_2001['location'] == most_wins_city)].sort_values('date').iloc[0]['score']

print(f"Final Answer: {most_wins_city}, {first_win}")