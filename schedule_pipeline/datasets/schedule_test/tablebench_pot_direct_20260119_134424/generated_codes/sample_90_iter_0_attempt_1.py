import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where song is "qyteti i dashurisë"
song_row = df[df['song'] == 'qyteti i dashurisë']
# Extract the points value
total_points = song_row['points'].values[0]
print(f"Final Answer: {total_points}")