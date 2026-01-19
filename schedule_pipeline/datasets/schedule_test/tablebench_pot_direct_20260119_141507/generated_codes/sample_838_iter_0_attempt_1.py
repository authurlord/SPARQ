import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the two schools and get their total medals
ayala_medals = df[df['ensemble'] == 'ayala high school']['total medals'].values[0]
franklin_medals = df[df['ensemble'] == 'franklin central hs']['total medals'].values[0]

if ayala_medals > franklin_medals:
    print("Final Answer: Ayala High School")
else:
    print("Final Answer: Franklin Central HS")