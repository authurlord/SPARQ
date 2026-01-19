import pandas as pd

df = pd.read_csv('table.csv')
# Find the beam type with the highest moment of inertia in torsion
max_j_index = df['moment of inertia in torsion (j) (cm 4 )'].idxmax()
beam_type = df.loc[max_j_index, 'type']
print(f"Final Answer: {beam_type}")