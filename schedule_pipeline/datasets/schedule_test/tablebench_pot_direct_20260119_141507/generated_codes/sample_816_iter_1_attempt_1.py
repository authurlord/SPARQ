import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Select only the judge columns (excluding 'artist', 'total', 'points')
judge_columns = [col for col in df.columns if col not in ['artist', 'total', 'points']]

# Convert all judge score columns to numeric, coercing errors to NaN
df_judges = df[judge_columns].apply(pd.to_numeric, errors='coerce')

# Compute standard deviation for each judge
std_deviation = df_judges.std()

# Find the judge with the highest standard deviation
max_std_judge = std_deviation.idxmax()

print(f"Final Answer: {max_std_judge}")