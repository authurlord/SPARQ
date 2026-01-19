import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'wins' column to numeric, excluding the 'total' row
wins = df[df['year'] != 'total']['wins'].astype(int)
# Calculate variance
variance_wins = wins.var()
print(f"Final Answer: {variance_wins:.2f}")