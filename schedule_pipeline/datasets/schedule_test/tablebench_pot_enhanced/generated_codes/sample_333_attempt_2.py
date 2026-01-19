import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Olympic Games and Javelin throw
olympic_javelin = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == 'Javelin throw')]
# Find the row with the best position (lowest numerical value in 'Position')
best_ranking = olympic_javelin.loc[olympic_javelin['Position'].astype(int).idxmin()]
print(f"Final Answer: {best_ranking['Year']}")