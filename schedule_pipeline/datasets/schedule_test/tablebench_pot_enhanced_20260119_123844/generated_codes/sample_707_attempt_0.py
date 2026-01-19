import pandas as pd

df = pd.read_csv('table.csv')
# Find the chambering with the highest p max
max_pressure_chambering = df.loc[df['p max'].idxmax(), 'chambering']
print(f"Final Answer: {max_pressure_chambering}")