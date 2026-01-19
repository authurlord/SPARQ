import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter for the 2006–07 season and Philadelphia team
filtered_df = df[(df['Year'] == '2006–07') & (df['Team'] == 'Philadelphia')]

# Extract the required stats
stats = filtered_df[['RPG', 'APG', 'SPG', 'BPG']].values.flatten()

# Define the labels for the radar chart
labels = ['Rebounds', 'Assists', 'Steals', 'Blocks']

# Create a radar chart
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
stats += stats[:1]  # Close the loop
angles += angles[:1]

# Plot
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

ax.fill(angles, stats, color='b', alpha=0.25)
ax.plot(angles, stats, color='b', linewidth=2)

# Set labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title('Average Rebounds, Assists, Steals, and Blocks per Game (2006–07, Philadelphia)', pad=20)
plt.show()

print(f"Final Answer: Radar chart for Philadelphia in 2006–07 season")