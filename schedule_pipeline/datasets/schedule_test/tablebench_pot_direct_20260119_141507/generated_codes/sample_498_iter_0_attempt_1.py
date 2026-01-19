import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Arabs' and 'total' to numeric, handling any potential formatting issues
df['arabs'] = pd.to_numeric(df['arabs'], errors='coerce')
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Calculate Arab demographic proportion as (Arabs / Total) * 100
df['arab_proportion'] = (df['arabs'] / df['total']) * 100

# Find the metropolitan ring with the highest Arab demographic proportion
max_proportion_ring = df.loc[df['arab_proportion'].idxmax(), 'metropolitan ring']

print(f"Final Answer: {max_proportion_ring}")