import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the winning team is 'united states'
us_wins = df[df['winning team'] == 'united states']
# Count how many times each US captain led the team
captain_counts = us_wins['us captain'].value_counts()
# Get the maximum count (most victories)
max_victories = captain_counts.max()
print(f"Final Answer: {max_victories}")