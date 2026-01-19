import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select relevant columns
votes = df[['county', 'obama', 'mccain']]

# Create a bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(votes))

# Plot Obama and McCain votes
bars1 = plt.bar([x - bar_width/2 for x in index], votes['obama'], bar_width, label='Obama', color='skyblue')
bars2 = plt.bar([x + bar_width/2 for x in index], votes['mccain'], bar_width, label='McCain', color='salmon')

# Labels and title
plt.xlabel('County')
plt.ylabel('Total Votes')
plt.title('Total Votes Received by Obama and McCain in Each County')
plt.xticks([x for x in index], votes['county'], rotation=45)

# Legend
plt.legend()

# Improve layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()