import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert all relevant columns to numeric, handling non-numeric entries like 'x', 'O' as NaN
df['Composition'] = pd.to_numeric(df['Composition'], errors='coerce')
df['Drawing'] = pd.to_numeric(df['Drawing'], errors='coerce')
df['Color'] = pd.to_numeric(df['Color'], errors='coerce')
df['Expression'] = pd.to_numeric(df['Expression'], errors='coerce')

# Get top 3 painters with highest Expression
top_3_composition = df.sort_values(by='Expression', ascending=False).head(3)['Composition'].mean()

# Get bottom 5 painters with lowest Color
bottom_5_drawing = df.sort_values(by='Color').head(5)['Drawing'].mean()

# Compute the difference
difference = top_3_composition - bottom_5_drawing

print(f"Final Answer: {difference:.1f}")