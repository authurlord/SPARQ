import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the standard deviation of 'points for'
std_points_for = df['points for'].std()
print(f"Final Answer: {std_points_for:.1f}")