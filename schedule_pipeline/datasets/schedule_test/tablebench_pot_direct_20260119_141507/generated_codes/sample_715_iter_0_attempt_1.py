import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'Composition' column to numeric, treating non-numeric values as NaN
df['Composition'] = pd.to_numeric(df['Composition'], errors='coerce')

# Find the painter with the highest Composition score
top_composition_painter = df.loc[df['Composition'].idxmax(), 'Painter']

print(f"Final Answer: {top_composition_painter}")