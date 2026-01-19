import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average of the 'points' column
average_points = df['points'].mean()
print(f"Final Answer: {average_points:.1f}")