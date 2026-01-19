import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Win' and 'Lose' to numeric, coercing errors to NaN
df['Win'] = pd.to_numeric(df['Win'], errors='coerce')
df['Lose'] = pd.to_numeric(df['Lose'], errors='coerce')

# Drop rows where either Win or Lose is NaN (invalid data)
df_clean = df.dropna(subset=['Win', 'Lose'])

# Extract the season from the 'League.1' column (e.g., '7th (2000-01)' → 2000-01)
df_clean['Season'] = df_clean['League.1'].str.extract(r'\((\d{4}-\d{2})\)')[0]

# Group by Season to get wins and losses
season_wins_losses = df_clean.groupby('Season')[['Win', 'Lose']].sum()

# Create a stacked bar chart
plt.figure(figsize=(10, 6))
season_wins_losses.plot(kind='bar', stacked=True, color=['skyblue', 'lightcoral'])
plt.title('Stacked Bar Chart of Wins and Losses by Season')
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Final Answer: The chart has been generated successfully
Final Answer: chart_generated