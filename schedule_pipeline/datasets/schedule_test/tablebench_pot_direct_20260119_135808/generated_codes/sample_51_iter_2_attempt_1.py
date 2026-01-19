import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Goal Difference is 16
team_with_goal_diff_16 = df[df['Goal Difference'] == 16]['Club']
# Print the result
print(f"Final Answer: {', '.join(team_with_goal_diff_16)}")