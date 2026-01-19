import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Select only the judge columns (excluding 'artist', 'total', 'points')
judge_columns = df.columns[1:-2]  # Skip 'artist' and 'total', 'points'

# Convert all judge score columns to numeric (int)
df_numeric = df[judge_columns].apply(pd.to_numeric, errors='coerce')

# Calculate standard deviation for each judge
std_deviation = df_numeric.std()

# Find the judge with the highest standard deviation
max_std_judge = std_deviation.idxmax()

print(f"Final Answer: {max_std_judge}")