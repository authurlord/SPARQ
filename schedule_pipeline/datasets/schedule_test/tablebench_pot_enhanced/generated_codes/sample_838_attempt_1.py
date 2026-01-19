import pandas as pd

df = pd.read_csv('table.csv')
# Filter the two schools
ayala_total = df[df['ensemble'] == 'ayala high school']['total medals'].values[0]
franklin_total = df[df['ensemble'] == 'franklin central hs']['total medals'].values[0]

# Compare and determine the higher one
if int(ayala_total) > int(franklin_total):
    winner = 'ayala high school'
else:
    winner = 'franklin central hs'

print(f"Final Answer: {winner}")