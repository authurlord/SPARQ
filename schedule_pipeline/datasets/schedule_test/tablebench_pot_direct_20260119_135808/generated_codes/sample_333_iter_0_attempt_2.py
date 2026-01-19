import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Olympic Games and Javelin throw events
olympic_javelin = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == 'Javelin throw')]
# Find the row with the best position (lowest position number)
best_ranking = olympic_javelin.loc[olympic_javelin['Position'].str.extract('(\d+)', expand=False).astype(int).idxmin()]
# Extract the year
year = best_ranking['Year']
print(f"Final Answer: {year}")