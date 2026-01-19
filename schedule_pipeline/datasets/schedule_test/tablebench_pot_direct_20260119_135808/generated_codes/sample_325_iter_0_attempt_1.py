import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Olympic Games and 1500 m event
olympic_1500m = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == '1500 m')]
# Find the best position (lowest rank)
best_position_row = olympic_1500m.loc[olympic_1500m['Position'].str.extract('(\d+)', expand=False).astype(int).idxmin()]
year = best_position_row['Year']
print(f"Final Answer: {year}")