import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter the row for English
english_data = df[df['Unnamed: 0'] == 'English'].iloc[0]
# Define the features and their values
features = ['agglutination', 'synthesis', 'compounding', 'derivation', 'inflection', 'prefixing', 'suffixing']
values = [float(english_data[feature]) for feature in features]

# Number of variables
num_vars = len(features)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

# Complete the loop
values += values[:1]
angles = np.concatenate((angles, [angles[0]]))

# Draw the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='red', alpha=0.25)
ax.plot(angles, values, color='red', linewidth=2)

# Add labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)
ax.set_rlabel_position(30)
ax.set_yticklabels([])

# Set title
plt.title("Language Features in English", size=16, color='red', pad=20)

# Show the plot
plt.show()