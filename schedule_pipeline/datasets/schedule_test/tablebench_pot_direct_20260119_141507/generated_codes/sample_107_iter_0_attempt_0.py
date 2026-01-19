import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter rows for ATL team and years 1997 to 2003
atl_data = df[(df['Team'] == 'ATL') & (df['Year'].astype(int) >= 1997) & (df['Year'].astype(int) <= 2003)]

# Extract relevant columns
atl_data = atl_data[['Year', 'Regular Season Won', 'Regular Season Lost', 'Regular Season Ties']]

# Convert Year to integer for proper sorting
atl_data['Year'] = atl_data['Year'].astype(int)

# Create a stacked bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.25
index = atl_data['Year'].values

# Plot wins, losses, and ties as stacked bars
wins = atl_data['Regular Season Won']
losses = atl_data['Regular Season Lost']
ties = atl_data['Regular Season Ties']

x_pos = range(len(index))

plt.bar(x_pos, wins, width=bar_width, label='Wins', color='skyblue')
plt.bar([p + bar_width for p in x_pos], losses, width=bar_width, label='Losses', color='salmon')
plt.bar([p + 2*bar_width for p in x_pos], ties, width=bar_width, label='Ties', color='lightgreen')

# Add total games played (wins + losses + ties) as a separate line
total_games = wins + losses + ties
plt.plot(x_pos, total_games, marker='o', color='black', linewidth=2, label='Total Games')

# Formatting
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.title('ATL Team Regular Season: Wins, Losses, Draws (1997-2003)')
plt.xticks([p + bar_width for p in x_pos], index, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()

# Final answer: Since the question asks for the chart, we don't return a numerical answer, but ensure the visualization is complete.
Final Answer: chart