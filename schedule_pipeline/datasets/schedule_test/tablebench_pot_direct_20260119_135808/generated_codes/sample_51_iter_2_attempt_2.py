import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Goal Difference' is 16
teams_with_16_goal_diff = df[df['Goal Difference'] == 16]['Club']
# Convert to list and print
print(f"Final Answer: {', '.join(teams_with_16_goal_diff)}")