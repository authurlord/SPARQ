import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean 'points for' column: split on space and take first number
df['points for'] = df['points for'].astype(str).str.split().str[0].astype(float)

# Calculate standard deviation of 'points for'
std_points_for = df['points for'].std()

print(f"Final Answer: {std_points_for:.1f}")