import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Drop rows where 'Drainage basin area [km2]' is NaN
df_clean = df.dropna(subset=['Drainage basin area [km2]'])

# Extract relevant columns
lengths = df_clean['Length [km]'].astype(float)
basin_areas = df_clean['Drainage basin area [km2]'].astype(float)

# Create a list of river names for x-axis labels
names = df_clean['Name']

# Create a figure and axis
plt.figure(figsize=(14, 8))

# Create grouped bar chart
x = range(len(names))
width = 0.35

plt.bar([i - width/2 for i in x], lengths, width, label='Length [km]', color='skyblue')
plt.bar([i + width/2 for i in x], basin_areas, width, label='Drainage basin area [km2]', color='lightcoral')

plt.xlabel('River Name')
plt.ylabel('Value')
plt.title('Length and Drainage Basin Area of Rivers')
plt.xticks([i for i in x], names, rotation=90)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()