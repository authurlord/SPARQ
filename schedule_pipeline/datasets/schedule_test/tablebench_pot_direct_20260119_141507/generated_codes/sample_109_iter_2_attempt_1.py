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

# Set up the figure and subplots
fig, ax = plt.subplots(figsize=(12, 8))

# Position of bars on x-axis
x = range(len(regions))

# Width of bars
bar_width = 0.2

# Create bars
bars1 = ax.bar([i - bar_width for i in x], populations, bar_width, label='Population', color='skyblue')
bars2 = ax.bar([i for i in x], areas, bar_width, label='Area (km²)', color='lightgreen')
bars3 = ax.bar([i + bar_width for i in x], densities, bar_width, label='Density (/km²)', color='salmon')

# Add labels and title
ax.set_xlabel('Region')
ax.set_ylabel('Value')
ax.set_title('Population, Area, and Density by Region')
ax.set_xticks(x)
ax.set_xticklabels(regions, rotation=45)

# Add value labels on top of bars
def add_value_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_value_labels(bars1, populations)
add_value_labels(bars2, areas)
add_value_labels(bars3, densities)

# Add legend
ax.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()