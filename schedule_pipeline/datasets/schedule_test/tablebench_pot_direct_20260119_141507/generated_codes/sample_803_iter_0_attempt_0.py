import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the last row (total) and calculate variance of 'wins'
wins_data = df[df['year'] != 'total']['wins'].dropna().astype(float)
variance_wins = wins_data.var()
print(f"Final Answer: {variance_wins:.2f}")