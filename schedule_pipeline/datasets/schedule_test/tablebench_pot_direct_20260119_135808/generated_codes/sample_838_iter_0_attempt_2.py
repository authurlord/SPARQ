import pandas as pd

df = pd.read_csv('table.csv')
# Filter the rows for the two schools
ayala_total = df[df['ensemble'] == 'ayala high school']['total medals'].values[0]
franklin_total = df[df['ensemble'] == 'franklin central hs']['total medals'].values[0]

# Compare and determine which has more medals
if ayala_total > franklin_total:
    winner = 'ayala high school'
else:
    winner = 'franklin central hs'

print(f"Final Answer: {winner}")