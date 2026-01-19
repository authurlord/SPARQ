import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Olympic Games
olympic_games = df[df['Competition'] == 'Olympic Games']
# Find the row with the best position (lowest numerical value in 'Position')
best_olympic_year = olympic_games.loc[olympic_games['Position'].astype(int).idxmin(), 'Year']
print(f"Final Answer: {best_olympic_year}")