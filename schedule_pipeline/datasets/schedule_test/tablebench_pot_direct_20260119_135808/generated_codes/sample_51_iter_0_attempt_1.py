import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Goal Difference' is 16
teams_with_goal_diff_16 = df[df['Goal Difference'] == 16]['Club']
# Output the result
print(f"Final Answer: {teams_with_goal_diff_16.values[0]}")