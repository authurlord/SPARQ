import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the 'total' row and convert 'wins' to numeric
wins = df[df['year'] != 'total']['wins'].astype(int)
# Calculate variance
variance_wins = wins.var()
print(f"Final Answer: {variance_wins:.4f}")