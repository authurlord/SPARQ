import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average points
average_points = df['points'].mean()
print(f"Final Answer: {average_points:.1f}")