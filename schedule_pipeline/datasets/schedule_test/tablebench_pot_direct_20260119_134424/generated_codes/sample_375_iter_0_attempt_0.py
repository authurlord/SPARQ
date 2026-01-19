import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where language is 'french' and count them
french_songs_count = df[df['language'] == 'french'].shape[0]
print(f"Final Answer: {french_songs_count}")