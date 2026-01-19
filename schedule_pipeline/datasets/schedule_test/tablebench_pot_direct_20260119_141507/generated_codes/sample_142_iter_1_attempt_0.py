import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'goal difference' to numeric (removing '+' sign)
df['goal difference'] = df['goal difference'].str.replace('+', '').astype(int)

# Sort by goal difference in descending order and take top 5
top_5 = df.nlargest(5, 'goal difference')[['club', 'goal difference']]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_5['club'], top_5['goal difference'], color='skyblue')
plt.title('Goal Difference of Top 5 Teams')
plt.xlabel('Team')
plt.ylabel('Goal Difference')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the result for verification
print(f"Final Answer: {top_5['club'].tolist()}, {top_5['goal difference'].tolist()}")