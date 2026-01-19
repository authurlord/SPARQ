import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Select top 5 teams
top_5_teams = df.head(5)
# Extract club names and goal differences
clubs = top_5_teams['club']
goal_diff = top_5_teams['goal difference']

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(clubs, goal_diff, color='skyblue')
plt.title('Goal Difference of Top 5 Teams')
plt.xlabel('Club')
plt.ylabel('Goal Difference')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()