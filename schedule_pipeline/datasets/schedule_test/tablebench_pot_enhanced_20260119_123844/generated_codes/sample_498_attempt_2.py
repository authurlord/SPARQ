import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' and 'arabs' columns to numeric
df['total'] = pd.to_numeric(df['total'])
df['arabs'] = pd.to_numeric(df['arabs'])

# Calculate Arab demographic proportion
df['arab_proportion'] = (df['arabs'] / df['total']) * 100

# Find the metropolitan ring with the highest Arab demographic proportion
max_ring = df.loc[df['arab_proportion'].idxmax(), 'metropolitan ring']

print(f"Final Answer: {max_ring}")