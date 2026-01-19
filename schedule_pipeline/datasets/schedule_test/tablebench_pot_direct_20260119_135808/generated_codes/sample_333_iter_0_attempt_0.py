import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Olympic Games
olympic_data = df[df['Competition'] == 'Olympic Games']
# Find the row with the best (lowest) position
best_olympic_year = olympic_data.loc[olympic_data['Position'].str.extract('(\d+)', expand=False).astype(int).idxmin(), 'Year']
print(f"Final Answer: {best_olympic_year}")