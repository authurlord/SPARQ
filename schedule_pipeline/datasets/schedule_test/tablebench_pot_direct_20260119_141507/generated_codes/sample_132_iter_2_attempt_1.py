import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant columns
df['Won'] = pd.to_numeric(df['Won'], errors='coerce')
df['Drawn'] = pd.to_numeric(df['Drawn'], errors='coerce')
df['Lost'] = pd.to_numeric(df['Lost'], errors='coerce')
df['Played'] = pd.to_numeric(df['Played'], errors='coerce')

# Ensure all values are valid
df = df.dropna(subset=['Won', 'Drawn', 'Lost', 'Played'])

# Calculate percentages
df['Win%'] = (df['Won'] / df['Played']) * 100
df['Draw%'] = (df['Drawn'] / df['Played']) * 100
df['Loss%'] = (df['Lost'] / df['Played']) * 100

# Prepare data for plotting
teams = df['Team']
win_percent = df['Win%']
draw_percent = df['Draw%']
loss_percent = df['Loss%']

# Create the stacked bar chart
plt.figure(figsize=(12, 8))
bar_width = 0.25
index = range(len(teams))

# Plot each percentage as a segment
plt.bar(index, win_percent, bar_width, label='Wins', color='skyblue')
plt.bar(index, draw_percent, bar_width, bottom=win_percent, label='Draws', color='lightgreen')
plt.bar(index, loss_percent, bar_width, bottom=win_percent + draw_percent, label='Losses', color='lightcoral')

# Customize the chart
plt.xlabel('Team')
plt.ylabel('Percentage (%)')
plt.title('Win-Loss-Draw Percentage by Team (Stacked Bar Chart)')
plt.xticks(index, teams, rotation=45)
plt.legend()

# Improve layout to prevent label cutoff
plt.tight_layout()

# Show the chart
plt.show()