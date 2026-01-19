import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' to numeric, coercing errors to NaN (but data seems clean)
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Sort by points in descending order and get the top 5 values
sorted_points = df.sort_values(by='points', ascending=False)['points']
highest_points = sorted_points.iloc[0]
fifth_highest_points = sorted_points.iloc[4]

# Calculate the difference
difference = highest_points - fifth_highest_points
print(f"Final Answer: {difference}")