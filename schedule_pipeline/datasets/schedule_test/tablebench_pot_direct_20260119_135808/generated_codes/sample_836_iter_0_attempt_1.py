import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'spanish' column for each municipality
spanish_speakers = df['spanish'].astype(int).sum()
# Find the municipality with the highest population of Spanish speakers
max_municipality = df.loc[df['spanish'].idxmax(), 'language']
print(f"Final Answer: {max_municipality}")