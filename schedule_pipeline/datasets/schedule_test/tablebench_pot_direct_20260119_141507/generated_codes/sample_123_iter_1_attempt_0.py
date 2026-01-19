import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Select relevant columns for the stacked bar chart
# We will use Wins, Draws, Losses as the components
data = df[['Wins', 'Draws', 'Losses']].copy()
teams = df['Club']

# Create a stacked bar chart
plt.figure(figsize=(12, 8))
bar_width = 0.25
index = range(len(teams))

# Plot each component (Wins, Draws, Losses) as a separate segment
bars = plt.bar(index, data['Wins'], bar_width, label='Wins', color='skyblue')
plt.bar([i + bar_width for i in index], data['Draws'], bar_width, label='Draws', color='lightgreen')
plt.bar([i + 2*bar_width for i in index], data['Losses'], bar_width, label='Losses', color='salmon')

# Customize the chart
plt.xlabel('Team')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Match Results (Wins, Draws, Losses)')
plt.xticks([i + bar_width for i in index], teams, rotation=45)
plt.legend()

# Improve layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()