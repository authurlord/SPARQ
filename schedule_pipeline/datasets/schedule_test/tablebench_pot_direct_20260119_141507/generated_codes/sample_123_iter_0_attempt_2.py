import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Parse the 'Points' column to extract numeric values
def parse_points(points_str):
    if '+' in points_str:
        parts = points_str.split('+')
        return int(parts[0]) + int(parts[1])
    else:
        return int(points_str)

df['Points'] = df['Points'].apply(parse_points)

# Extract relevant columns
teams = df['Club']
wins = df['Wins'].astype(int)
draws = df['Draws'].astype(int)
losses = df['Losses'].astype(int)

# Create a figure and axis
plt.figure(figsize=(12, 8))

# Create grouped bar chart
x = range(len(teams))
width = 0.2

# Bar positions
bars1 = plt.bar([p - width/2 for p in x], wins, width, label='Wins', color='skyblue')
bars2 = plt.bar([p + width/2 for p in x], draws, width, label='Draws', color='lightgreen')
bars3 = plt.bar([p + width/2 for p in x], losses, width, label='Losses', color='salmon')

# Labels and title
plt.xlabel('Teams')
plt.ylabel('Number of Matches')
plt.title('Match Statistics: Wins, Draws, and Losses per Team')
plt.xticks(x, teams, rotation=45)

# Add legend
plt.legend()

# Improve layout
plt.tight_layout()

# Show plot
plt.show()