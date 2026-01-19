import pandas as pd

df = pd.read_csv('table.csv')
# Find the beam type with the highest moment of inertia in torsion
max_inertia_row = df.loc[df['moment of inertia in torsion (j) (cm 4 )'].idxmax()]
beam_type = max_inertia_row['type']
print(f"Final Answer: {beam_type}")