import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter for the 2006–07 season and Philadelphia team
filtered_data = df[(df['Year'] == '2006–07') & (df['Team'] == 'Philadelphia')]

# Extract the required stats
stats = filtered_data[['RPG', 'APG', 'SPG', 'BPG']].iloc[0].values

# Prepare labels
labels = ['Rebounds', 'Assists', 'Steals', 'Blocks']

# Number of variables
num_vars = len(labels)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Complete the loop
stats += stats[:1]  # repeat first value to close the radar chart
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, stats, color='b', alpha=0.25)
ax.plot(angles, stats, color='b', linewidth=2, linestyle='solid')

# Add labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title('Average Rebounds, Assists, Steals, and Blocks per Game (2006–07, Philadelphia)', pad=20)
plt.show()

# Final Answer: The radar chart is generated with the average stats for the Philadelphia player in 2006–07.
Final Answer: 2006–07, Philadelphia