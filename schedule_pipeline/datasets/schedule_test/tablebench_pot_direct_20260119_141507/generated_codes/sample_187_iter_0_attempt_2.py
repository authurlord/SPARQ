import pandas as pd

df = pd.read_csv('table.csv')

# Calculate expected points based on wins and draws
df['expected_points'] = 3 * df['wins'] + df['draws']

# Compare expected points with actual points
point_difference = df['expected_points'] - df['points']

# Check if the difference is close to zero (within a small tolerance)
tolerance = 1e-5
is_consistent = point_difference.abs().min() < tolerance

if is_consistent:
    final_answer = "wins, draws"
else:
    final_answer = "no clear impact"

print(f"Final Answer: {final_answer}")