import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert population and area to integers
df['Population (2003 est.)'] = df['Population (2003 est.)'].str.replace(',', '').astype(int)
df['Area (km²)'] = df['Area (km²)'].astype(int)
df['Density (/km²)'] = df['Density (/km²)'].astype(int)

# Prepare data for plotting
regions = df['Name']
populations = df['Population (2003 est.)']
areas = df['Area (km²)']
densities = df['Density (/km²)']

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Create grouped bars
bar_width = 0.2
index = range(len(regions))
ax.bar([i - bar_width for i in index], populations, bar_width, label='Population', color='skyblue')
ax.bar([i for i in index], areas, bar_width, label='Area (km²)', color='lightgreen')
ax.bar([i + bar_width for i in index], densities, bar_width, label='Density (/km²)', color='salmon')

# Customize the plot
ax.set_xlabel('Region')
ax.set_ylabel('Value')
ax.set_title('Population, Area, and Density by Region')
ax.set_xticks(index)
ax.set_xticklabels(regions, rotation=45)
ax.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()