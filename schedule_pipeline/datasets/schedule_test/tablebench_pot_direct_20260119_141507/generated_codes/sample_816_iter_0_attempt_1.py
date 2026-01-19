import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Select only the judge columns (excluding 'artist', 'total', 'points')
judge_columns = [col for col in df.columns if col not in ['artist', 'total', 'points']]
# Calculate standard deviation for each judge
std_devs = df[judge_columns].apply(lambda x: np.std(x, ddof=1))
# Find the judge with the highest standard deviation
most_variable_judge = std_devs.idxmax()
print(f"Final Answer: {most_variable_judge}")