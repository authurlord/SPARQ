import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where narrator is 'katherine kellgren'
kellgren_books = df[df['narrator'] == 'katherine kellgren']
# Check if any of these books won in 2013
winner_2013 = kellgren_books[(kellgren_books['citation'] == 'winner') & (kellgren_books['year'] == '2013')]
# Extract the author
author = winner_2013['author'].iloc[0] if not winner_2013.empty else None
print(f"Final Answer: {author}")