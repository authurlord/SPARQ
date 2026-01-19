import pandas as pd
import numpy as np

# Load the table
df = pd.read_csv('table.csv')

# Convert all relevant columns to numeric, replacing non-numeric values like 'O', 'x' with NaN
df[['Composition', 'Drawing', 'Color', 'Expression']] = df[['Composition', 'Drawing', 'Color', 'Expression']].apply(pd.to_numeric, errors='coerce')

# Step 1: Top 3 painters with highest 'Expression' scores
top_3_expr = df.nlargest(3, 'Expression')[['Composition']]

# Average 'Composition' of top 3
avg_composition_top3 = top_3_expr['Composition'].mean()

# Step 2: Bottom 5 painters with lowest 'Color' scores
bottom_5_color = df.nsmallest(5, 'Color')[['Drawing']]

# Average 'Drawing' of bottom 5
avg_drawing_bottom5 = bottom_5_color['Drawing'].mean()

# Calculate the difference
difference = avg_composition_top3 - avg_drawing_bottom5

print(f"Final Answer: {difference:.1f}")