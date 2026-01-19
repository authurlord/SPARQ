import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter data for English
english_data = df[df['Unnamed: 0'] == 'English'].iloc[0]

# Define the features and their values
features = ['agglutination', 'synthesis', 'compounding', 'derivation', 'inflection', 'prefixing', 'suffixing']
values = [float(english_data[feat]) for feat in features]

# Calculate angle for each feature
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)

# Close the loop
values += values[:1]
angles += [angles[0]]

# Plot radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values, linewidth=2, linestyle='solid', label='English')
ax.fill(angles, values, alpha=0.25)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=10)
ax.set_title("Language Features in English", size=14, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.show()