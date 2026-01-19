import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the two schools and get their total medals
ayala_medals = df[df['ensemble'] == 'ayala high school']['total medals'].values[0]
franklin_medals = df[df['ensemble'] == 'franklin central hs']['total medals'].values[0]

# Compare and output the school with higher total medals
if ayala_medals > franklin_medals:
    final_answer = "Ayala High School"
else:
    final_answer = "Franklin Central HS"

print(f"Final Answer: {final_answer}")