import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert population, area, and density to numeric (remove commas and parse)
df['Population (2003 est.)'] = df['Population (2003 est.)'].str.replace(',', '').astype(int)
df['Area (km²)'] = df['Area (km²)'].astype(int)
df['Density (/km²)'] = df['Density (/km²)'].astype(int)

# Create subplots for each metric
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

# Population bar chart
axes[0].bar(df['Name'], df['Population (2003 est.)'])
axes[0].set_title('Population (2003 est.)')
axes[0].set_xlabel('Region')
axes[0].set_ylabel('Population')
axes[0].tick_params(axis='x', rotation=45)

# Area bar chart
axes[1].bar(df['Name'], df['Area (km²)'])
axes[1].set_title('Area (km²)')
axes[1].set_xlabel('Region')
axes[1].set_ylabel('Area (km²)')
axes[1].tick_params(axis='x', rotation=45)

# Density bar chart
axes[2].bar(df['Name'], df['Density (/km²)'])
axes[2].set_title('Density (/km²)')
axes[2].set_xlabel('Region')
axes[2].set_ylabel('Density (/km²)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()