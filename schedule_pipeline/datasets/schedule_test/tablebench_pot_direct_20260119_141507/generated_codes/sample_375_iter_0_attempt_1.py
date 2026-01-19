import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of songs where language is 'french'
french_songs_count = df[df['language'] == 'french'].shape[0]
print(f"Final Answer: {french_songs_count}")