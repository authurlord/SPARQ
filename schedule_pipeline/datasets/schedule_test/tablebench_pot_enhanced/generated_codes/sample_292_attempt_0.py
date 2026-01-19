import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Sunny Morning in 2017
sunny_morning_2017 = df[(df['Nominated Work'] == 'Sunny Morning') & (df['Year'] == '2017')]

# Extract the award and notes
award = sunny_morning_2017['Award'].values[0]
notes = sunny_morning_2017['Notes'].values[0]

# Extract edition number from notes
edition = None
if 'edition' in notes:
    edition = notes.split('edition #')[1].split(' ')[0]

print(f"Final Answer: Radio Music Awards, {edition}")