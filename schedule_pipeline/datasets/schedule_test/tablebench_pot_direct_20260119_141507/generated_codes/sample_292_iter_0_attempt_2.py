import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Nominated Work is 'Sunny Morning' and Year is 2017
filtered_rows = df[(df['Nominated Work'] == 'Sunny Morning') & (df['Year'] == '2017')]

# Extract the award and notes (edition) for the Hollywood Music in Media Awards
award_won = filtered_rows['Award'].iloc[0]
# Find the Hollywood Music in Media Awards row with "Sunny Morning" and extract edition from Notes
hmm_row = df[(df['Nominated Work'] == 'Sunny Morning') & (df['Award'] == 'Hollywood Music in Media Awards')]
edition = hmm_row['Notes'].iloc[0] if not hmm_row.empty else None

print(f"Final Answer: {award_won}, {edition}")