import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select relevant columns
votes = df[['county', 'obama', 'mccain']]

# Create a bar chart
plt.figure(figsize=(12, 6))
x = votes['county']
obama_votes = votes['obama']
mccain_votes = votes['mccain']

# Create grouped bar chart
bar_width = 0.35
index = range(len(x))
plt.bar([i - bar_width/2 for i in index], obama_votes, bar_width, label='Obama', color='skyblue')
plt.bar([i + bar_width/2 for i in index], mccain_votes, bar_width, label='McCain', color='salmon')

# Labels and title
plt.xlabel('County')
plt.ylabel('Total Votes')
plt.title('Total Votes Received by Obama and McCain in Each County')
plt.xticks([i for i in index], x, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Display the chart
plt.show()