import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Arabs' and 'total' to numeric, then compute proportion
df['arab_proportion'] = (pd.to_numeric(df['arabs'], errors='coerce') / pd.to_numeric(df['total'], errors='coerce')) * 100

# Find the row with the highest Arab proportion
max_arab_ring = df.loc[df['arab_proportion'].idxmax(), 'metropolitan ring']

print(f"Final Answer: {max_arab_ring}")