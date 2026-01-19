import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'spanish' column for each municipality
spanish_speakers = df.set_index('language')['spanish'].astype(int).sum()
# Find the municipality with the highest Spanish speaker count
max_municipality = df.set_index('language')['spanish'].astype(int).idxmax()
print(f"Final Answer: {max_municipality}")