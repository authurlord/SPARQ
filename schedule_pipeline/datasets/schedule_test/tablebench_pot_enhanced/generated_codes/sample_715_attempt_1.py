import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Composition' column to numeric, coercing errors to NaN
df['Composition'] = pd.to_numeric(df['Composition'], errors='coerce')
# Find the painter with the highest Composition score
max_composition_painter = df.loc[df['Composition'].idxmax(), 'Painter']
print(f"Final Answer: {max_composition_painter}")