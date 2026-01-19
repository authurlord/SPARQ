import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' to numeric, coercing errors to NaN (should not happen here)
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Sort by points in descending order and take top 3
top_3 = df.sort_values(by='points', ascending=False).head(3)

# Calculate average of top 3 points
avg_top_3 = top_3['points'].mean()

# Get the highest point value
max_points = df['points'].max()

# Compute the difference
difference = max_points - avg_top_3

print(f"Final Answer: {difference:.0f}")