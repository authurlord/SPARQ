import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert the relevant columns to numeric (in case of any formatting issues)
df['Won'] = pd.to_numeric(df['Won'], errors='coerce')
df['Drawn'] = pd.to_numeric(df['Drawn'], errors='coerce')
df['Lost'] = pd.to_numeric(df['Lost'], errors='coerce')
df['Played'] = pd.to_numeric(df['Played'], errors='coerce')

# Calculate percentages
df['Win%'] = (df['Won'] / df['Played']) * 100
df['Draw%'] = (df['Drawn'] / df['Played']) * 100
df['Loss%'] = (df['Lost'] / df['Played']) * 100

# Prepare data for plotting
teams = df['Team']
win_percentages = df['Win%']
draw_percentages = df['Draw%']
loss_percentages = df['Loss%']

# Create a figure and axis
plt.figure(figsize=(12, 8))

# Create stacked bar chart
bars = plt.bar(teams, win_percentages, label='Win', color='skyblue')
plt.bar(teams, draw_percentages, bottom=win_percentages, label='Draw', color='lightgreen')
plt.bar(teams, loss_percentages, bottom=win_percentages + draw_percentages, label='Loss', color='salmon')

# Customize the plot
plt.xlabel('Team')
plt.ylabel('Percentage (%)')
plt.title('Win-Loss-Draw Percentages by Team (Stacked Bar Chart)')
plt.legend()
plt.xticks(rotation=45)

# Improve layout
plt.tight_layout()

# Display the chart
plt.show()