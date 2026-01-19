import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select the row for English
english_data = df[df['Unnamed: 0'] == 'English'].iloc[0].drop('Unnamed: 0').values

# Define the features (columns)
features = ['agglutination', 'synthesis', 'compounding', 'derivation', 'inflection', 'prefixing', 'suffixing']
labels = features

# Convert to float
english_values = [float(x) for x in english_data]

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()

# Complete the polygon by repeating the first value
english_values += english_values[:1]
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, english_values, color='b', alpha=0.25)
ax.plot(angles, english_values, color='b', linewidth=2)

# Set labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Add title
plt.title('Language Features in English', pad=20)

# Show the plot
plt.show()