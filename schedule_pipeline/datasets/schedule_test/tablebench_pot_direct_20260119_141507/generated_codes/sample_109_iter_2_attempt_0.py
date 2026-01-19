import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert population, area, and density to numeric
df['Population (2003 est.)'] = pd.to_numeric(df['Population (2003 est.)'].str.replace(',', ''), errors='coerce')
df['Area (km²)'] = pd.to_numeric(df['Area (km²)'], errors='coerce')
df['Density (/km²)'] = pd.to_numeric(df['Density (/km²)'], errors='coerce')

# Prepare data for plotting
regions = df['Name']
populations = df['Population (2003 est.)']
areas = df['Area (km²)']
densities = df['Density (/km²)']

# Set up the bar chart
fig, ax = plt.subplots(figsize=(12, 7))

x = range(len(regions))
width = 0.2

# Create bars for each metric
ax.bar([p - width for p in x], populations, width, label='Population', color='skyblue')
ax.bar([p for p in x], areas, width, label='Area (km²)', color='lightgreen')
ax.bar([p + width for p in x], densities, width, label='Density (/km²)', color='salmon')

# Labels and title
ax.set_xlabel('Region')
ax.set_ylabel('Value')
ax.set_title('Population, Area, and Density by Region')
ax.set_xticks(x)
ax.set_xticklabels(regions, rotation=45)
ax.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()