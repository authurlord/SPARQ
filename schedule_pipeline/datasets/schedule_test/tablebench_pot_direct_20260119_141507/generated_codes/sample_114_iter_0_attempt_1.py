import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter for the 2006–07 season and Philadelphia team
filtered_data = df[(df['Year'] == '2006–07') & (df['Team'] == 'Philadelphia')]

# Extract the required metrics
metrics = ['RPG', 'APG', 'SPG', 'BPG']
values = filtered_data[metrics].iloc[0].dropna().values

# Create radar chart
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
values += values[:1]  # Close the loop
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='b', alpha=0.25)
ax.plot(angles, values, color='b', linewidth=2)

# Add labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)

plt.title('Average Rebounds, Assists, Steals, and Blocks per Game (2006–07, Philadelphia)', pad=20)
plt.show()

print("Final Answer: Radar chart generated for Philadelphia in 2006–07 season")