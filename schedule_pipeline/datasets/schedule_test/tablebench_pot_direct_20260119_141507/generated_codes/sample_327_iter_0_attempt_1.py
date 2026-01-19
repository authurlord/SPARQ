import pandas as pd

df = pd.read_csv('table.csv')

# Filter books where narrator is 'Katherine Kellgren'
kellgren_books = df[df['narrator'] == 'Katherine Kellgren']

# Get the authors of those books
kellgren_authors = kellgren_books['author'].tolist()

# Filter books from 2013 with citation 'winner'
winning_2013 = df[(df['year'] == '2013') & (df['citation'] == 'winner')]

# Get authors of winning 2013 books
winning_2013_authors = winning_2013['author'].tolist()

# Find intersection
common_authors = set(kellgren_authors) & set(winning_2013_authors)

if common_authors:
    final_answer = list(common_authors)[0]
else:
    final_answer = "none"

print(f"Final Answer: {final_answer}")