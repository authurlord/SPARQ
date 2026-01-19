import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract top 5 teams by goal difference (descending order)
top_5 = df.sort_values(by='goal difference', ascending=False).head(5)

# Convert 'goal difference' from string to numeric (remove '+' and convert to int)
top_5['goal_difference_numeric'] = top_5['goal difference'].str.replace('+', '').astype(int)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_5['club'], top_5['goal_difference_numeric'], color='skyblue')
plt.xlabel('Club')
plt.ylabel('Goal Difference')
plt.title('Goal Differences of Top 5 Teams')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

# Print the goal differences for clarity (optional)
print(f"Final Answer: {top_5['goal_difference_numeric'].tolist()}")