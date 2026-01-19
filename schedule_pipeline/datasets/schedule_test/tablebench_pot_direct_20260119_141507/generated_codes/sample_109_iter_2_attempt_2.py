import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert population and area to numeric
df['Population (2003 est.)'] = pd.to_numeric(df['Population (2003 est.)'].str.replace(',', ''), errors='coerce')
df['Area (km²)'] = pd.to_numeric(df['Area (km²)'], errors='coerce')
df['Density (/km²)'] = pd.to_numeric(df['Density (/km²)'], errors='coerce')

# Select relevant columns
regions = df['Name']
populations = df['Population (2003 est.)']
areas = df['Area (km²)']
densities = df['Density (/km²)']

# Create a figure with subplots for each metric
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

# Population bar chart
ax[0].bar(regions, populations, color='skyblue')
ax[0].set_title('Population (2003 est.)')
ax[0].set_xlabel('Region')
ax[0].set_ylabel('Population')

# Area bar chart
ax[1].bar(regions, areas, color='lightgreen')
ax[1].set_title('Area (km²)')
ax[1].set_xlabel('Region')
ax[1].set_ylabel('Area (km²)')

# Density bar chart
ax[2].bar(regions, densities, color='salmon')
ax[2].set_title('Density (/km²)')
ax[2].set_xlabel('Region')
ax[2].set_ylabel('Density (/km²)')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()