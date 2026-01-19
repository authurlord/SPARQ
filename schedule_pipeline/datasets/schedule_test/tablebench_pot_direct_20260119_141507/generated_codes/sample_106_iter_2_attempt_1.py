import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select the row for English
english_row = df[df['Unnamed: 0'] == 'English'].iloc[0]

# Extract feature names (excluding 'Unnamed: 0')
features = ['agglutination', 'synthesis', 'compounding', 'derivation', 'inflection', 'prefixing', 'suffixing']
values = [
    float(english_row['agglutination']),
    float(english_row['synthesis']),
    float(english_row['compounding']),
    float(english_row['derivation']),
    float(english_row['inflection']),
    float(english_row['prefixing']),
    float(english_row['suffixing'])
]

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()

# Repeat the first value to close the circle
values += values[:1]
angles += angles[:1]

# Create radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='b', alpha=0.25)
ax.plot(angles, values, color='b', linewidth=2)

# Add labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)

# Set title
plt.title('Language Features in English', pad=20)

# Show plot
plt.show()