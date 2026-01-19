import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for Guercino
guercino_data = df[df['Painter'] == 'Guercino'].iloc[0]
values = [guercino_data['Composition'], guercino_data['Drawing'], guercino_data['Color'], guercino_data['Expression']]
labels = ['Composition', 'Drawing', 'Color', 'Expression']

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# Complete the polygon by repeating the first value
values += values[:1]
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='b', alpha=0.25)
ax.plot(angles, values, color='b', linewidth=2)

# Set labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Add title
plt.title('Performance of Guercino', pad=20)

# Show the plot
plt.show()

# Final Answer: The radar chart has been generated successfully.
Final Answer: radar_chart