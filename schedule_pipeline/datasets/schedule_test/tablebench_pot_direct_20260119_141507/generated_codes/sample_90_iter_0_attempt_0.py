import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where 'song' is 'qyteti i dashurisë'
song_points = df[df['song'] == 'qyteti i dashurisë']['points'].values[0]
print(f"Final Answer: {song_points}")