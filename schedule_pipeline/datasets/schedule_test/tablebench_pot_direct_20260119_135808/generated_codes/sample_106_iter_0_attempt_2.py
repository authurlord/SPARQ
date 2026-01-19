import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter data for English
english_data = df[df['Unnamed: 0'] == 'English'].iloc[0]

# Define the features (language processes)
features = ['agglutination', 'synthesis', 'compounding', 'derivation', 'inflection', 'prefixing', 'suffixing']

# Extract values for English
values = [float(english_data[feature]) for feature in features]

# Number of variables
num_vars = len(features)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

# Close the circular graph
values += values[:1]
angles += angles[:1]

# Plot radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values, linewidth=2, linestyle='solid', label='English')
ax.fill(angles, values, alpha=0.25)

# Set labels for each axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)

# Set y-axis limits and labels
ax.set_rlabel_position(30)
ax.set_yticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5'])

# Add title and legend
plt.title("Language Features in English", size=16, color='blue', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Show the plot
plt.show()