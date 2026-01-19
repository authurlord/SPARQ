import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Clean the season from League.1 column (e.g., '7th (2000-01)' -> '2000-01')
df['season'] = df['League.1'].str.extract(r'\((\d{4}-\d{2})\)')[0].fillna('Unknown')

# Drop rows where season is missing
df = df.dropna(subset=['season'])

# Convert 'Win' and 'Lose' to integers
df['Win'] = pd.to_numeric(df['Win'], errors='coerce')
df['Lose'] = pd.to_numeric(df['Lose'], errors='coerce')

# Group by season and sum wins and losses
season_data = df.groupby('season')[['Win', 'Lose']].sum().reset_index()

# Plot stacked bar chart
plt.figure(figsize=(10, 6))
plt.bar(season_data['season'], season_data['Win'], label='Wins', color='skyblue')
plt.bar(season_data['season'], season_data['Lose'], bottom=season_data['Win'], label='Losses', color='lightcoral')
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins and Losses by Season')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final Answer: The chart has been generated as requested.
Final Answer: chart_generated