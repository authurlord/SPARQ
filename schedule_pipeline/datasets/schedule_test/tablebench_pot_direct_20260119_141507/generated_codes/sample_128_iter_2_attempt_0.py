import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter data for Guercino
guercino_data = df[df['Painter'] == 'Guercino'].iloc[0]

# Extract the values for the radar chart
categories = ['Composition', 'Drawing', 'Color', 'Expression']
values = [guercino_data['Composition'], guercino_data['Drawing'], guercino_data['Color'], guercino_data['Expression']]

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

# Complete the loop
values += values[:1]
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='skyblue', alpha=0.5)
ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')

# Add labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Title and legend
plt.title('Performance of Guercino', pad=20)
plt.show()

# Final answer
print("Final Answer: Guercino")