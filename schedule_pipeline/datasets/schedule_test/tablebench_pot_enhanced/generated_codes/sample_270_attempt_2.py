import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where result is 'final'
final_artists = df[df['result'] == 'final']
# Sum the jury votes for these artists
total_jury_votes = final_artists['jury votes'].sum()
print(f"Final Answer: {total_jury_votes}")