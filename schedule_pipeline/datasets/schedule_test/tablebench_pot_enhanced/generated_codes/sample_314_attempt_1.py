import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Grammy Awards and Song of the Year category
song_of_year_nominations = df[(df['Association'] == 'Grammy Awards') & (df['Category'] == 'Song of the Year')]

# Get the year(s) where the song was nominated
nominated_years = song_of_year_nominations['Year'].tolist()

# Check if in any of these years, the same work won another award
for year in nominated_years:
    # Get the nominated work for that year
    nominated_work = song_of_year_nominations[song_of_year_nominations['Year'] == year]['Nominated work'].iloc[0]
    
    # Check if this work won any award in the same year
    wins_in_year = df[(df['Year'] == year) & (df['Nominated work'] == nominated_work) & (df['Result'] == 'Won')]
    
    if not wins_in_year.empty:
        print(f"Final Answer: {year}")
        break