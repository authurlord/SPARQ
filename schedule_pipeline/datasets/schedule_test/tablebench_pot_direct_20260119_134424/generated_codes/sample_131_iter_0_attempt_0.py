import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Combine 'League' and 'League.1' to form a unique label for each season
df['season_label'] = df['League'] + ' ' + df['League.1']

# Set up the plot
plt.figure(figsize=(12, 6))
bars1 = plt.bar(df['season_label'], df['Win'], label='Wins', color='skyblue')
bars2 = plt.bar(df['season_label'], df['Lose'], bottom=df['Win'], label='Losses', color='lightcoral')

# Add labels and title
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.title('Wins and Losses by Season')
plt.xticks(rotation=45, ha='right')

# Add legend
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()