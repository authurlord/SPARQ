import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where narrator is 'katherine kellgren'
kellgren_books = df[df['narrator'] == 'katherine kellgren']

# Check if any of these authors have a book that won in 2013
winning_2013 = df[(df['citation'] == 'winner') & (df['year'] == '2013')]

# Find the intersection of authors
matching_author = set(kellgren_books['author']).intersection(set(winning_2013['author']))

# Output the result
print(f"Final Answer: {matching_author.pop()}")