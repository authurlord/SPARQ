import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select the row for English
english_row = df[df['Unnamed: 0'] == 'English'].iloc[0]

# List of features
features = ['agglutination', 'synthesis', 'compounding', 'derivation', 'inflection', 'prefixing', 'suffixing']
values = [float(x) for x in english_row[features]]

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()

# Close the plot by appending the first value
values += values[:1]
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='b', alpha=0.25)
ax.plot(angles, values, color='b', linewidth=2)

# Add labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)

plt.title('Language Features in English', pad=20)
plt.show()