import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['Population (2003 est.)'] = pd.to_numeric(df['Population (2003 est.)'].str.replace(',', ''), errors='coerce')
df['Area (km²)'] = pd.to_numeric(df['Area (km²)'], errors='coerce')
df['Density (/km²)'] = pd.to_numeric(df['Density (/km²)'], errors='coerce')

# Select the columns for plotting
metrics = ['Population (2003 est.)', 'Area (km²)', 'Density (/km²)']
names = df['Name']

# Create subplots for each metric
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

for i, metric in enumerate(metrics):
    axes[i].bar(names, df[metric], color=plt.cm.viridis(i / 3))
    axes[i].set_title(f'{metric}')
    axes[i].set_xlabel('Region')
    axes[i].set_ylabel('Value')
    plt.suptitle('Population, Area, and Density by Region', fontsize=16)

plt.tight_layout()
plt.show()