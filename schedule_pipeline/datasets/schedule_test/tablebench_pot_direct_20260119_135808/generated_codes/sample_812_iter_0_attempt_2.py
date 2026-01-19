import pandas as pd

df = pd.read_csv('table.csv')
# Remove the header row and convert 'points for' to numeric
df = df[1:]
df['points for'] = df['points for'].str.replace(' ', '').astype(int)
# Calculate standard deviation
std_points_for = df['points for'].std()
print(f"Final Answer: {std_points_for:.2f}")