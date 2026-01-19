import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Grammy Awards, Song of the Year category, and Won result
winning_song_of_year = df[(df['Association'] == 'Grammy Awards') & 
                          (df['Category'] == 'Song of the Year') & 
                          (df['Result'] == 'Won')]

# Extract the year
year = winning_song_of_year['Year'].iloc[0]
print(f"Final Answer: {year}")