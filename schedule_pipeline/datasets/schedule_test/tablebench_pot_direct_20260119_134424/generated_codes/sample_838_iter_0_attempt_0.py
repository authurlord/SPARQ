import pandas as pd

df = pd.read_csv('table.csv')
# Filter the two schools
ayala_medals = df[df['ensemble'] == 'ayala high school']['total medals'].values[0]
franklin_medals = df[df['ensemble'] == 'franklin central hs']['total medals'].values[0]

# Compare and determine the higher one
if ayala_medals > franklin_medals:
    winner = 'ayala high school'
else:
    winner = 'franklin central hs'

print(f"Final Answer: {winner}")