import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to numeric, coercing errors to NaN and then dropping invalid entries
df['points'] = pd.to_numeric(df['points'], errors='coerce')
# Drop any rows with invalid points (in case of error)
df = df.dropna(subset=['points'])

# Sort by points in descending order and take top 3
top_3 = df.nlargest(3, 'points')

# Calculate average of top 3 points
avg_top_3 = top_3['points'].mean()

# Get the highest point value
max_points = df['points'].max()

# Compute the difference
difference = max_points - avg_top_3

print(f"Final Answer: {difference:.0f}")