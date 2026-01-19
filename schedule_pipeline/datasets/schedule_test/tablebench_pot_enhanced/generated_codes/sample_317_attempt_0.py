import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the winning team is 'united states'
us_wins = df[df['winning team'] == 'united states']
# Count how many times each US captain led the team to victory
captain_victories = us_wins['us captain'].value_counts()
# Get the maximum number of victories by a US captain
max_victories = captain_victories.max()
print(f"Final Answer: {max_victories}")