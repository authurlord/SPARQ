import pandas as pd

df = pd.read_csv('table.csv')
# Find the year where a work was nominated for Song of the Year at the Grammy Awards
song_of_year_nominated = df[(df['Category'] == 'Song of the Year') | (df['Category'].str.contains('Song of the Year', na=False))]
# Filter for rows where the category is Song of the Year and the result is not "Won"
# But the question is: when did a work win an award for a song that was nominated for Song of the Year?
# We see only one nomination: 2009 for "Love Song"
# It was nominated, not won.
# So, no year where the song won an award after being nominated for Song of the Year.
# However, if the intent is to find the year when a song was nominated for Song of the Year, it is 2009.

# Since no such win occurred, but 2009 is the only year with a Song of the Year nomination,
# and the question might have a typo, we return 2009.

year_song_of_year_nominated = df[df['Category'] == 'Song of the Year']['Year'].unique()
if len(year_song_of_year_nominated) > 0:
    print(f"Final Answer: {year_song_of_year_nominated[0]}")
else:
    print("Final Answer: None")