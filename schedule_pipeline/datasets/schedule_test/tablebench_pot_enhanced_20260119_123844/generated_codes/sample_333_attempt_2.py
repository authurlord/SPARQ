import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Olympic Games and Javelin throw events
olympic_javelin = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == 'Javelin throw')]
# Find the row with the best (lowest) position
best_position_row = olympic_javelin.loc[olympic_javelin['Position'].str.extract('(\d+)').astype(int).idxmin()]
# Extract the year
year = best_position_row['Year']
print(f"Final Answer: {year}")