import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter data for ATL team from 1997 to 2003
atl_data = df[(df['Team'] == 'ATL') & (df['Year'].astype(int).between(1997, 2003))]

# Prepare the data for plotting
years = atl_data['Year'].values
wins = atl_data['Regular Season Won'].values
losses = atl_data['Regular Season Lost'].values
ties = atl_data['Regular Season Ties'].values

# Total games per year
total_games = wins + losses + ties

# Create the stacked bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(years))

# Create bars for wins, losses, and ties (stacked)
bars1 = plt.bar(index, wins, bar_width, label='Wins', color='skyblue')
bars2 = plt.bar(index, losses, bar_width, label='Losses', color='salmon', bottom=wins)
bars3 = plt.bar(index, ties, bar_width, label='Ties', color='lightgreen', bottom=wins + losses)

# Add total games as a separate line plot (optional, for clarity)
plt.plot(index, total_games, marker='o', color='red', linestyle='--', label='Total Games', linewidth=2)

# Labels and title
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.title('ATL Team Regular Season Performance (1997-2003)')
plt.xticks(index, years)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Print total games for reference
print(f"Final Answer: {total_games}")