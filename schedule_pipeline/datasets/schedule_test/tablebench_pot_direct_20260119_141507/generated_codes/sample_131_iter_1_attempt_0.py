import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Parse the season from League.1
def extract_season(league_entry):
    if isinstance(league_entry, str):
        # Try to extract year from format like '7th (2000-01)'
        import re
        match = re.search(r'\((\d{4}-\d{2})\)|(\d{4}-\d{2})', league_entry)
        if match:
            return match.group(1)
        # If no match, try direct format like '2006-07'
        if '-' in league_entry and len(league_entry.split('-')) == 2:
            return league_entry.split('-')[0] + '-' + league_entry.split('-')[1]
    return None

df['season'] = df['League.1'].apply(extract_season)

# Filter out rows where season is not extracted properly
df = df.dropna(subset=['season'])

# Convert 'Win' and 'Lose' to numeric (they are already integers)
df['Win'] = pd.to_numeric(df['Win'], errors='coerce')
df['Lose'] = pd.to_numeric(df['Lose'], errors='coerce')

# Group by season and sum wins and losses
season_wins_losses = df.groupby('season')[['Win', 'Lose']].sum().reset_index()

# Plot stacked bar chart
plt.figure(figsize=(10, 6))
plt.bar(season_wins_losses['season'], season_wins_losses['Win'], label='Wins', color='skyblue')
plt.bar(season_wins_losses['season'], season_wins_losses['Lose'], bottom=season_wins_losses['Win'], label='Losses', color='lightcoral')
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins and Losses by Season')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final Answer: The chart is displayed as requested
Final Answer: chart