import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Maesteg RFC
maesteg_row = df[df['club'] == 'maesteg rfc']
# Extract the 'played' value
games_played = maesteg_row['played'].values[0]
print(f"Final Answer: {games_played}")