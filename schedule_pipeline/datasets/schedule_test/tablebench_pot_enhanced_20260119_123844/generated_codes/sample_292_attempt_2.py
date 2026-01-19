import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Sunny Morning in 2017 with a win
sunny_morning_win = df[(df['Nominated Work'] == 'Sunny Morning') & (df['Year'] == '2017') & (df['Result'] == 'Won')]
# Extract award name
award_name = sunny_morning_win['Award'].values[0]
# Extract edition from notes for Hollywood Music in Media Awards
hmi_notes = sunny_morning_win[sunny_morning_win['Award'] == 'Hollywood Music in Media Awards']['Notes'].values[0]
edition = hmi_notes.split('edition #')[1].split(' ')[0]

print(f"Final Answer: {award_name}, {edition}")