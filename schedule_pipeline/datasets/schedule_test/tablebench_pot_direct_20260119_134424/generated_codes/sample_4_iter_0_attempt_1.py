import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool and Premier League seasons
liverpool_premier = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]
# Sum the goals scored in Premier League (League_1 column)
total_goals = liverpool_premier['League_1'].sum()
print(f"Final Answer: {total_goals}")