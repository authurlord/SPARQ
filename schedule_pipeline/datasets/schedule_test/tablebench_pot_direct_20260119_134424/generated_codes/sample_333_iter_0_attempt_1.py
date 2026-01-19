import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Olympic Games and Javelin throw
olympic_javelin = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == 'Javelin throw')]
# Find the row with the best position (lowest rank)
best_rank_row = olympic_javelin.loc[olympic_javelin['Position'].str.replace('th', '').str.replace('nd', '').str.replace('rd', '').str.replace('st', '').astype(int).idxmin()]
# Extract the year
year = best_rank_row['Year']
print(f"Final Answer: {year}")