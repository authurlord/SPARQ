import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total votes' and 'televotes' to numeric, coercing errors to NaN if any
df['total votes'] = pd.to_numeric(df['total votes'], errors='coerce')
df['televotes'] = pd.to_numeric(df['televotes'], errors='coerce')

# Calculate Tele Efficiency
df['Tele Efficiency'] = df['total votes'] / df['televotes']

# Find the artist with the highest Tele Efficiency
max_efficiency_artist = df.loc[df['Tele Efficiency'].idxmax(), 'artist']

print(f"Final Answer: {max_efficiency_artist}")