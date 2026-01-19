import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select relevant columns
length_km = df['Length [km]']
basin_area = df['Drainage basin area [km2]']
names = df['Name']

# Create a figure and axis
plt.figure(figsize=(14, 8))

# Create bar positions
x = range(len(names))
width = 0.35

# Plot length and basin area bars
plt.bar([i - width/2 for i in x], length_km, width, label='Length [km]', color='skyblue')
plt.bar([i + width/2 for i in x], basin_area, width, label='Drainage basin area [km2]', color='lightcoral')

# Labels and title
plt.xlabel('River Name')
plt.ylabel('Value')
plt.title('Length of Rivers and Their Drainage Basin Areas')
plt.xticks([i for i in x], names, rotation=90)

# Legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()