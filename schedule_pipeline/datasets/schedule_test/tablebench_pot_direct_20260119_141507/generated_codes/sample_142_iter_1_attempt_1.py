import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'goal difference' column to numeric, handling strings like '+34'
df['goal difference'] = df['goal difference'].str.replace('+', '').str.replace('-', '').astype(float)
df['goal difference'] = df['goal difference'].apply(lambda x: x if x >= 0 else -x)  # Ensure positive values for visualization

# Sort by goal difference in descending order and take top 5
top_5 = df.sort_values(by='goal difference', ascending=False).head(5)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_5['club'], top_5['goal difference'], color='skyblue')
plt.xlabel('Club')
plt.ylabel('Goal Difference')
plt.title('Goal Difference of Top 5 Teams')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the goal differences for clarity (optional output)
print(f"Final Answer: {top_5['goal difference'].tolist()}")