import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Get the top 5 teams by position (first 5 rows)
top_5_teams = df.head(5)

# Extract team names and goal difference
teams = top_5_teams['club']
goal_diff = top_5_teams['goal difference']

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(teams, goal_diff, color='skyblue')
plt.xlabel('Club')
plt.ylabel('Goal Difference')
plt.title('Goal Differences of Top 5 Teams')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the chart
plt.show()