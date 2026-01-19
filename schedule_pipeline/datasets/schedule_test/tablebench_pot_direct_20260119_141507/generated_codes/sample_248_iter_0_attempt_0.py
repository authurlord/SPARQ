import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Replace 'O', 'x' with NaN and convert to numeric
df['Expression'] = pd.to_numeric(df['Expression'], errors='coerce')
df['Color'] = pd.to_numeric(df['Color'], errors='coerce')

# Top 3 painters with highest Expression
top_3_expr = df.sort_values(by='Expression', ascending=False).head(3)
avg_composition_top3 = top_3_expr['Composition'].mean()

# Bottom 5 painters with lowest Color
bottom_5_color = df.sort_values(by='Color').head(5)
avg_drawing_bottom5 = bottom_5_color['Drawing'].mean()

# Calculate the difference
difference = avg_composition_top3 - avg_drawing_bottom5
print(f"Final Answer: {difference:.1f}")