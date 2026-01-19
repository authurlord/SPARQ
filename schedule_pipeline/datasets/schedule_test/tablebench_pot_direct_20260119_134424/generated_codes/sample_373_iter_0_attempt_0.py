import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool and Premier League seasons
liverpool_premier = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]
# Count the number of seasons
num_seasons = len(liverpool_premier)
print(f"Final Answer: {num_seasons}")