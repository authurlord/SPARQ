import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Select only the judge columns (excluding 'artist', 'total', 'points')
judge_columns = [col for col in df.columns if col not in ['artist', 'total', 'points']]
# Compute standard deviation for each judge
std_devs = df[judge_columns].std()
# Find the judge with the highest standard deviation
max_std_judge = std_devs.idxmax()
print(f"Final Answer: {max_std_judge}")