import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert population and area to numeric
df['Population (2003 est.)'] = pd.to_numeric(df['Population (2003 est.)'].str.replace(',', ''), errors='coerce')
df['Area (km²)'] = pd.to_numeric(df['Area (km²)'], errors='coerce')

# Select the columns for plotting
columns_to_plot = ['Population (2003 est.)', 'Area (km²)', 'Density (/km²)']
region_names = df['Name']

# Create subplots for each metric
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

for i, col in enumerate(columns_to_plot):
    axes[i].bar(region_names, df[col], color=plt.cm.viridis(i / len(columns_to_plot)))
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('Region')
    axes[i].set_ylabel('Value')
    plt.setp(axes[i].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()