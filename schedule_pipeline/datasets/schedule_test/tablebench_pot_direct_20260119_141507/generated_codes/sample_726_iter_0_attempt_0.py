import pandas as pd

df = pd.read_csv('table.csv')
# Find the beam type with the highest moment of inertia in torsion
max_torsion = df.loc[df['moment of inertia in torsion (j) (cm 4 )'].idxmax(), 'type']
print(f"Final Answer: {max_torsion}")