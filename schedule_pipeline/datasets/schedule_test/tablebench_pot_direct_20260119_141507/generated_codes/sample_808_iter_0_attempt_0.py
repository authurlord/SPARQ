import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'P' column to string and remove non-numeric entries
df['P'] = df['P'].astype(str)
# Filter rows where 'P' contains only digits (i.e., purely numeric)
df_numeric = df[df['P'].str.isdigit()]
# Extract numeric values from 'P' column
numeric_values = df_numeric['P'].astype(int)
# Calculate mean
mean_p = numeric_values.mean()
print(f"Final Answer: {mean_p:.1f}")