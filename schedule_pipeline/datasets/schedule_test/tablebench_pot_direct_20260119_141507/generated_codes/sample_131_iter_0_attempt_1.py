import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert the data to a clean format for plotting
# We'll extract the 'Win' and 'Lose' columns and keep the season information
# The 'League' and 'Position' are used to identify the season, but we'll just use the row as a season

# Create a list of seasons based on the first column (League) and the position
seasons = []
wins = []
losses = []

for idx, row in df.iterrows():
    season_info = row['League'] + ' - ' + str(row['Position']).strip()
    seasons.append(season_info)
    wins.append(row['Win'])
    losses.append(row['Lose'])

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'Season': seasons,
    'Wins': wins,
    'Losses': losses
})

# Plot stacked bar chart
plt.figure(figsize=(12, 6))
plot_df.set_index('Season').plot(kind='bar', stacked=True, color=['skyblue', 'lightcoral'])
plt.title('Stacked Bar Chart of Wins and Losses by Season')
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.xticks(rotation=45)
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()

# Final Answer: The chart has been generated showing wins and losses per season.
Final Answer: chart_generated