import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the total row (last row) and calculate variance of wins
df_filtered = df[df['year'] != 'total']
wins = df_filtered['wins'].astype(float)
variance_wins = wins.var()
print(f"Final Answer: {variance_wins:.2f}")