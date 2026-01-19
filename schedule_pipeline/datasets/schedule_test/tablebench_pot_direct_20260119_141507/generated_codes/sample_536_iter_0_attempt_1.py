import pandas as pd

df = pd.read_csv('table.csv')
# Calculate Tele Efficiency: total votes / televotes
df['tele_efficiency'] = df['total votes'] / df['televotes']
# Find the artist with the highest tele efficiency
top_artist = df.loc[df['tele_efficiency'].idxmax(), 'artist']
print(f"Final Answer: {top_artist}")