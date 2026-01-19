import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' to numeric, coercing errors to NaN (should not happen here)
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Get the highest and second-highest points
highest_points = df['points'].max()
second_highest_points = df['points'].nlargest(2).iloc[1]  # Second largest value

# Calculate the difference
difference = highest_points - second_highest_points
print(f"Final Answer: {difference}")