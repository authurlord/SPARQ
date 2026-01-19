import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Goal Difference' is 16
teams_with_goal_diff_16 = df[df['Goal Difference'] == 16]['Club']
# Convert to list and print the result
print(f"Final Answer: {', '.join(teams_with_goal_diff_16)}")