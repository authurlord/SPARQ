import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter for the 2006–07 season and Philadelphia team
filtered_data = df[(df['Year'] == '2006–07') & (df['Team'] == 'Philadelphia')]

# Extract the stats: RPG, APG, SPG, BPG
stats = filtered_data[['RPG', 'APG', 'SPG', 'BPG']].values[0]

# Define the categories
categories = ['Rebounds', 'Assists', 'Steals', 'Blocks']

# Create a radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

# Repeat the first value to close the circle
stats += stats[:1]
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, stats, color='b', alpha=0.25)
ax.plot(angles, stats, color='b', linewidth=2, linestyle='solid')

# Add labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

plt.title('Average Rebounds, Assists, Steals, and Blocks per Game (2006–07, Philadelphia)', pad=20)
plt.show()