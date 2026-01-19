import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Select relevant columns for the chart
columns_to_use = ['Position', 'Wins', 'Draws', 'Losses']
df_filtered = df[columns_to_use]

# Convert Wins, Draws, Losses to integers
df_filtered['Wins'] = pd.to_numeric(df_filtered['Wins'], errors='coerce')
df_filtered['Draws'] = pd.to_numeric(df_filtered['Draws'], errors='coerce')
df_filtered['Losses'] = pd.to_numeric(df_filtered['Losses'], errors='coerce')

# Drop any rows with NaN (in case of parsing issues)
df_filtered = df_filtered.dropna()

# Create a stacked bar chart
plt.figure(figsize=(12, 8))
bar_width = 0.25
positions = df_filtered['Position'].astype(str).values
wins = df_filtered['Wins'].values
draws = df_filtered['Draws'].values
losses = df_filtered['Losses'].values

# Create bars
x = range(len(positions))
plt.bar(x, wins, width=bar_width, label='Wins', edgecolor='black')
plt.bar([p + bar_width for p in x], draws, width=bar_width, label='Draws', edgecolor='black')
plt.bar([p + 2*bar_width for p in x], losses, width=bar_width, label='Losses', edgecolor='black')

# Customize labels and title
plt.xlabel('Team Position')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins, Draws, and Losses by Team Position')
plt.xticks([p + bar_width for p in x], positions, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()