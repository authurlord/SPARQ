import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select relevant columns
votes = df[['county', 'obama', 'mccain']]

# Create a bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(votes))

# Bar positions
obama_bars = [x for x in index]
mccain_bars = [x + bar_width for x in index]

# Plot bars
plt.bar(obama_bars, votes['obama'], bar_width, label='Obama', color='skyblue')
plt.bar(mccain_bars, votes['mccain'], bar_width, label='McCain', color='salmon')

# Labels and title
plt.xlabel('County')
plt.ylabel('Total Votes')
plt.title('Total Votes Received by Obama and McCain in Each County')
plt.xticks([x + bar_width / 2 for x in index], votes['county'], rotation=45)

# Legend
plt.legend()

# Improve layout
plt.tight_layout()

# Show plot
plt.show()