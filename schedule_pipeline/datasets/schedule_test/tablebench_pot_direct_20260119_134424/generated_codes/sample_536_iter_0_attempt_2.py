import pandas as pd

df = pd.read_csv('table.csv')
# Calculate Tele Efficiency
df['tele_efficiency'] = df['total votes'] / df['televotes']
# Find the artist with the highest Tele Efficiency
max_efficiency_artist = df.loc[df['tele_efficiency'].idxmax(), 'artist']
print(f"Final Answer: {max_efficiency_artist}")