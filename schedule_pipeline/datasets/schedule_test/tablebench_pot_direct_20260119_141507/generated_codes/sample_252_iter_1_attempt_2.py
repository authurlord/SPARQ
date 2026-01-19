import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' to numeric, coercing errors to NaN
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Remove any rows with invalid points (if any)
df = df.dropna(subset=['points'])

# Sort by points in descending order and get top 3
top_3 = df.nlargest(3, 'points')

# Calculate average points of top 3
avg_top_3 = top_3['points'].mean()

# Find the highest point value
max_points = df['points'].max()

# Compute the difference
difference = max_points - avg_top_3

print(f"Final Answer: {difference:.0f}")