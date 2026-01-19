import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter for ATL team from 1997 to 2003, excluding aggregate rows
filtered_df = df[(df['Team'] == 'ATL') & (df['Year'].astype(int).between(1997, 2003))]

# Extract relevant columns
wins = filtered_df['Regular Season Won']
losses = filtered_df['Regular Season Lost']
ties = filtered_df['Regular Season Ties']

# Create a list of years
years = filtered_df['Year'].astype(int)

# Prepare data for stacked bar chart
data = {
    'Wins': wins,
    'Losses': losses,
    'Ties': ties
}

# Create the stacked bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.25
index = range(len(years))

# Plot each category
bars = []
for i, (key, values) in enumerate(data.items()):
    bars.append(plt.bar([x + i * bar_width for x in index], values, bar_width, label=key))

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.title('Regular Season Wins, Losses, and Ties for ATL (1997–2003)')
plt.xticks([x + bar_width for x in index], years)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Final Answer: We have generated the chart as requested.
Final Answer: chart_generated