import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter for 2006–07 season and Philadelphia team
filtered_data = df[(df['Year'] == '2006–07') & (df['Team'] == 'Philadelphia')]

# Extract the required stats
stats = filtered_data[['RPG', 'APG', 'SPG', 'BPG']].iloc[0].values

# Create radar chart
labels = ['Rebounds (RPG)', 'Assists (APG)', 'Steals (SPG)', 'Blocks (BPG)']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# Close the plot
stats += stats[:1]  # Complete the loop
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, stats, color='b', alpha=0.25)
ax.plot(angles, stats, color='b', linewidth=2)

# Set labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Add title
plt.title('Average Rebounds, Assists, Steals, and Blocks per Game (2006–07, Philadelphia)', pad=20)

# Show plot
plt.show()

# Final Answer: The radar chart is generated for the specified season and team.
Final Answer: radar_chart