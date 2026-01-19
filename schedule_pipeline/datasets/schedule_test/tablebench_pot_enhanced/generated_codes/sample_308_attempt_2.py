import pandas as pd

df = pd.read_csv('table.csv')
# Filter for songs by Glen Campbell
glen_songs = df[df['Artist'] == 'Glen Campbell']
# Filter for songs released in the 1980s
glen_1980s_songs = glen_songs[glen_songs['Year'].astype(int) >= 1980]
glen_1980s_songs = glen_1980s_songs[glen_1980s_songs['Year'].astype(int) <= 1989]
# Count the number of songs
count = len(glen_1980s_songs)
print(f"Final Answer: {count}")