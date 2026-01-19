import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Olympic Games and 1500 m event
olympic_1500m = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == '1500 m')]
# Find the best position (lowest number)
best_position_row = olympic_1500m.loc[olympic_1500m['Position'].str.replace('th', '').str.replace('sf', '').str.replace('h', '').astype(int).idxmin()]
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")