import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool in Premier League seasons
liverpool_premier = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]
# Sum the 'Goals' column for these rows
total_goals = liverpool_premier['Goals'].sum()
print(f"Final Answer: {total_goals}")