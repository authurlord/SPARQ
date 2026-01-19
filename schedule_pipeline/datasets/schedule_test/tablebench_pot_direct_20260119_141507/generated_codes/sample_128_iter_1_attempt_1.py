import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for Guercino
guercino_row = df[df['Painter'] == 'Guercino'].iloc[0]
values = guercino_row[['Composition', 'Drawing', 'Color', 'Expression']].fillna(0).astype(float)

# Define the categories
categories = ['Composition', 'Drawing', 'Color', 'Expression']

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

# Repeat the first value to close the circle
values += values[:1]
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='skyblue', alpha=0.5)
ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')

# Set category labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add title
plt.title('Performance of Guercino', size=16, pad=20)

# Show the plot
plt.show()

# Final Answer: The radar chart has been successfully generated for Guercino's performance.
Final Answer: radar_chart