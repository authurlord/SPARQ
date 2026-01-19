import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert Winners and Runner-ups to integers (some values are strings)
df['Winners'] = pd.to_numeric(df['Winners'], errors='coerce')
df['Runner-ups'] = pd.to_numeric(df['Runner-ups'], errors='coerce')

# Calculate total top positions (wins + runner-ups)
df['Total Top Positions'] = df['Winners'] + df['Runner-ups']

# Prepare data for plotting
clubs = df['Club']
total_positions = df['Total Top Positions']

# Create bar chart
plt.figure(figsize=(12, 6))
plt.bar(clubs, total_positions, color='skyblue')
plt.xlabel('Club')
plt.ylabel('Total Top Positions (Wins + Runner-ups)')
plt.title('Number of Times Each Club Has Won First or Second Place')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

# Print the total top positions for reference
print(f"Final Answer: {dict(zip(clubs, total_positions))}")