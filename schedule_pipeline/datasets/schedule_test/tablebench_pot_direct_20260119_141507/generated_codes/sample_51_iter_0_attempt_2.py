import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Goal Difference' is exactly 16
teams_with_diff_16 = df[df['Goal Difference'] == 16]['Club']
print(f"Final Answer: {', '.join(teams_with_diff_16.tolist())}")