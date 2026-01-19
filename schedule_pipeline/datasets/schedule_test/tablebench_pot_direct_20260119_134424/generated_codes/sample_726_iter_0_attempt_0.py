import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum value in 'moment of inertia in torsion (j) (cm 4 )'
max_inertia_row = df.loc[df['moment of inertia in torsion (j) (cm 4 )'].idxmax()]
beam_type = max_inertia_row['type']
print(f"Final Answer: {beam_type}")