import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where language is 'french'
french_songs = df[df['language'] == 'french']
# Count the number of French songs
count_french_songs = len(french_songs)
print(f"Final Answer: {count_french_songs}")