import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Extract English row (index 5)
english_data = df[df['Unnamed: 0'] == 'English'].iloc[0].drop('Unnamed: 0').values

# Define the features (columns)
features = ['agglutination', 'synthesis', 'compounding', 'derivation', 'inflection', 'prefixing', 'suffixing']

# Convert to numeric (some values are strings with decimals)
english_values = [float(x) for x in english_data]

# Number of variables
num_vars = len(features)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Complete the loop
angles += angles[:1]  # Close the circle

# Values for English
english_values += english_values[:1]

# Create radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, english_values, color='skyblue', alpha=0.5)
ax.plot(angles, english_values, color='blue', linewidth=2, linestyle='-', marker='o')

# Add labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)

# Title
plt.title('Language Features in English', pad=20)

# Show the plot
plt.show()