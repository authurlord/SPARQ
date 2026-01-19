import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Get the top 5 teams by points (first 5 rows since they are sorted by position)
top_5 = df.head(5)

# Extract team names and goal differences
teams = top_5['club'].tolist()
goal_diff = top_5['goal difference'].astype(float).tolist()

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(teams, goal_diff, color='skyblue')
plt.xlabel('Team')
plt.ylabel('Goal Difference')
plt.title('Goal Differences of Top 5 Teams')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the goal differences for reference
print(f"Final Answer: {goal_diff}")