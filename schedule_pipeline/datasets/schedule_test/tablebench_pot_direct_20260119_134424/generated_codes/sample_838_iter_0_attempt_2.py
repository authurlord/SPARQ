import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Ayala High School and Franklin Central HS
ayala_medals = df[df['ensemble'] == 'ayala high school']['total medals'].values[0]
franklin_medals = df[df['ensemble'] == 'franklin central hs']['total medals'].values[0]

# Compare and determine the school with higher total medals
if ayala_medals > franklin_medals:
    winner = 'ayala high school'
else:
    winner = 'franklin central hs'

print(f"Final Answer: {winner}")