import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where club is 'maesteg rfc' and get the 'played' value
maesteg_played = df[df['club'] == 'maesteg rfc']['played'].values[0]
print(f"Final Answer: {maesteg_played}")