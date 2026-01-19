import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Select top 8 teams (rank 1 to 8)
top_8 = df.head(8)

# Prepare data for stacked bar chart
data = top_8[['Won', 'Drawn', 'Lost']]
labels = top_8['Club']

# Create stacked bar chart
plt.figure(figsize=(12, 6))
data.plot(kind='bar', stacked=True, color=['green', 'yellow', 'red'])
plt.title('Wins, Draws, and Losses for Top 8 Ranked Teams')
plt.xlabel('Team')
plt.ylabel('Number of Matches')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()