import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert population, area, and density to numeric
df['Population (2003 est.)'] = pd.to_numeric(df['Population (2003 est.)'].str.replace(',', ''), errors='coerce')
df['Area (km²)'] = pd.to_numeric(df['Area (km²)'], errors='coerce')
df['Density (/km²)'] = pd.to_numeric(df['Density (/km²)'], errors='coerce')

# Select names and values for plotting
names = df['Name']
populations = df['Population (2003 est.)']
areas = df['Area (km²)']
densities = df['Density (/km²)']

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

# Population bar chart
axs[0].bar(names, populations, color='skyblue')
axs[0].set_title('Population (2003 est.)')
axs[0].set_xlabel('Region')
axs[0].set_ylabel('Population')

# Area bar chart
axs[1].bar(names, areas, color='lightgreen')
axs[1].set_title('Area (km²)')
axs[1].set_xlabel('Region')
axs[1].set_ylabel('Area (km²)')

# Density bar chart
axs[2].bar(names, densities, color='salmon')
axs[2].set_title('Density (/km²)')
axs[2].set_xlabel('Region')
axs[2].set_ylabel('Density (/km²)')

# Adjust layout and show plot
plt.tight_layout()
plt.show()