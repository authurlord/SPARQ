import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Goal Difference is exactly 16
teams = df[df['Goal Difference'] == '16']['Club']
# Convert to list and print
print(f"Final Answer: {', '.join(teams.tolist())}")