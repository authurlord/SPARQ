import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where 'Nominated Work' is 'Sunny Morning'
sunny_morning_rows = df[df['Nominated Work'] == 'Sunny Morning']

# Extract the award won in 2017 (Radio Music Awards)
award_won = sunny_morning_rows[sunny_morning_rows['Year'] == '2017']['Award'].iloc[0]

# Extract the edition from Hollywood Music in Media Awards nomination
hmm_edition = df[(df['Nominated Work'] == 'Sunny Morning') & (df['Award'] == 'Hollywood Music in Media Awards')]['Notes'].iloc[0]

print(f"Final Answer: {award_won}, {hmm_edition}")