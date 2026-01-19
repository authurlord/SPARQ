import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for 2006-07 season and Philadelphia team
filtered_row = df[(df['Year'] == '2006–07') & (df['Team'] == 'Philadelphia')]

# Extract the required metrics
metrics = filtered_row[['RPG', 'APG', 'SPG', 'BPG']].values[0]

# Create a radar chart
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
metrics += metrics[:1]  # Close the loop
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, metrics, color='b', alpha=0.25)
ax.plot(angles, metrics, color='b', linewidth=2)

# Label the axes
labels = ['Rebounds', 'Assists', 'Steals', 'Blocks']
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title('Average Rebounds, Assists, Steals, and Blocks per Game (2006–07, Philadelphia)')
plt.show()

# Final Answer: The radar chart has been successfully generated for the specified season and team.
Final Answer: radar_chart_generated