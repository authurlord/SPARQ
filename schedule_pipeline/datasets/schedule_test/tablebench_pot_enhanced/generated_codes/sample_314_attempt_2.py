import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Nominated work' is 'Love Song' and 'Category' is 'Song of the Year'
song_of_year_nominee = df[(df['Nominated work'] == 'Love Song') & (df['Category'] == 'Song of the Year')]
# Get the year when this nomination occurred
year_nomination = song_of_year_nominee['Year'].iloc[0]

# Now check if 'Love Song' won any award
winning_award = df[(df['Nominated work'] == 'Love Song') & (df['Result'] == 'Won')]
# Get the year of the win
year_win = winning_award['Year'].iloc[0]

print(f"Final Answer: {year_win}")