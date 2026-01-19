import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Arabs' and 'total' to numeric and calculate Arab proportion
df['arab_proportion'] = (pd.to_numeric(df['arabs'], errors='coerce') / pd.to_numeric(df['total'], errors='coerce')) * 100

# Find the row with the highest Arab proportion
max_arab_row = df.loc[df['arab_proportion'].idxmax()]
highest_arab_ring = max_arab_row['metropolitan ring']

print(f"Final Answer: {highest_arab_ring}")