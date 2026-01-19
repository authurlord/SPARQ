import pandas as pd

df = pd.read_csv('table.csv')
# The column 'points' contains numerical values
mean_points = df['points'].mean()
print(f"Final Answer: {mean_points:.1f}")