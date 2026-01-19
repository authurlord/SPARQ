import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Find Guercino's row
guercino_row = df[df['Painter'] == 'Guercino'].iloc[0]
values = [guercino_row['Composition'], guercino_row['Drawing'], guercino_row['Color'], guercino_row['Expression']]

# Labels for the radar chart
labels = ['Composition', 'Drawing', 'Color', 'Expression']

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# Complete the radar chart by repeating the first value
values += values[:1]
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='b', alpha=0.25)
ax.plot(angles, values, color='b', linewidth=2)

# Add labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Title
plt.title('Performance of Guercino', pad=20)

# Show the chart
plt.show()

# Final Answer: The radar chart has been successfully displayed.
Final Answer: radar_chart