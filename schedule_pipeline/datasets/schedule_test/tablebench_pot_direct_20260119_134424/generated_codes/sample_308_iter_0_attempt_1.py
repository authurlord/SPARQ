import pandas as pd

df = pd.read_csv('table.csv')
# Filter for songs by Glen Campbell
glen_campbell_songs = df[df['Artist'] == 'Glen Campbell']
# Filter for songs released in the 1980s
eighties_songs = glen_campbell_songs[glen_campbell_songs['Year'].astype(int) >= 1980]
eighties_songs = eighties_songs[eighties_songs['Year'].astype(int) <= 1989]
# Count the number of songs
count = len(eighties_songs)
print(f"Final Answer: {count}")