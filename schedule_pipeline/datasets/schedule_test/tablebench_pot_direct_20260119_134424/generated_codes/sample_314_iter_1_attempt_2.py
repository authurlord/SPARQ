import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Association is 'Grammy Awards' and Category is 'Song of the Year'
song_of_the_year_nominations = df[(df['Association'] == 'Grammy Awards') & (df['Category'] == 'Song of the Year')]
# Find the row where the result is 'Won'
winner_row = song_of_the_year_nominations[song_of_the_year_nominations['Result'] == 'Won']
# Extract the Year
year = winner_row['Year'].iloc[0]
print(f"Final Answer: {year}")