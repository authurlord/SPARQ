import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Composition' column to numeric, treating non-numeric values as NaN
df['Composition'] = pd.to_numeric(df['Composition'], errors='coerce')
# Find the painter with the maximum Composition score
max_composition = df['Composition'].max()
painter_with_max_composition = df[df['Composition'] == max_composition]['Painter'].iloc[0]
print(f"Final Answer: {painter_with_max_composition}")