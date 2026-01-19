import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Olympic Games
olympic_games = df[df['Competition'] == 'Olympic Games']
# Find the row with the best (lowest) position
best_position_row = olympic_games.loc[olympic_games['Position'].str.extract('(\d+)', expand=False).astype(int).idxmin()]
# Extract the year
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")