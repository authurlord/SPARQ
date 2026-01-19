import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Exclude the 'total' row and convert 'wins' to integers
wins = df[df['year'] != 'total']['wins'].astype(int)
# Calculate variance
variance_wins = np.var(wins, ddof=1)  # Using sample variance (ddof=1)
print(f"Final Answer: {variance_wins:.2f}")