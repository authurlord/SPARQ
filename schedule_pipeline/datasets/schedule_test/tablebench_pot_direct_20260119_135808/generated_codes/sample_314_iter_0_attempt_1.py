import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Grammy Awards, Song of the Year, Nominated
song_of_year_nominated = df[(df['Association'] == 'Grammy Awards') & 
                             (df['Category'] == 'Song of the Year') & 
                             (df['Result'] == 'Nominated')]

# Get the year when "Love Song" was nominated
nominated_year = song_of_year_nominated['Year'].iloc[0]

# Now check if "Love Song" won any award in any year
love_song_wins = df[(df['Nominated work'] == 'Love Song') & (df['Result'] == 'Won')]

# If there is a win, return the year of the win
if not love_song_wins.empty:
    winning_year = love_song_wins['Year'].iloc[0]
    print(f"Final Answer: {winning_year}")
else:
    print(f"Final Answer: {nominated_year}")