import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where narrator is 'katherine kellgren'
kellgren_books = df[df['narrator'] == 'katherine kellgren']
# Get the authors of these books
kellgren_authors = kellgren_books['author'].unique()

# Check which of these authors wrote a winning book in 2013
winning_2013 = df[(df['year'] == '2013') & (df['citation'] == 'winner')]
winning_authors_2013 = winning_2013['author'].unique()

# Find intersection of authors
matching_authors = set(kellgren_authors).intersection(set(winning_authors_2013))

print(f"Final Answer: {matching_authors.pop()}")