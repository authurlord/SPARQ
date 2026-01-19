import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Select top 8 teams (first 8 rows)
top_8 = df.iloc[0:8]

# Extract wins, draws, and losses
wins = top_8['Won'].values
draws = top_8['Drawn'].values
losses = top_8['Lost'].values

# Create a list of team names from the 'Club' column
teams = top_8['Club'].values

# Create the stacked bar chart
plt.figure(figsize=(12, 6))
x_positions = range(len(teams))
width = 0.25

# Create bars for wins, draws, and losses
plt.bar(x_positions, wins, width, label='Wins', color='skyblue')
plt.bar([p + width for p in x_positions], draws, width, label='Draws', color='lightgreen')
plt.bar([p + 2*width for p in x_positions], losses, width, label='Losses', color='salmon')

# Customize the chart
plt.xlabel('Team')
plt.ylabel('Number of Matches')
plt.title('Wins, Draws, and Losses for Top 8 Ranked Teams')
plt.xticks([p + width for p in x_positions], teams, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()