import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total votes' and 'televotes' to numeric, coercing errors to NaN if any
df['total votes'] = pd.to_numeric(df['total votes'], errors='coerce')
df['televotes'] = pd.to_numeric(df['televotes'], errors='coerce')

# Calculate Tele Efficiency (total_votes / televotes)
df['tele_efficiency'] = df['total votes'] / df['televotes']

# Find the artist with the highest tele efficiency
best_artist = df.loc[df['tele_efficiency'].idxmax(), 'artist']
print(f"Final Answer: {best_artist}")