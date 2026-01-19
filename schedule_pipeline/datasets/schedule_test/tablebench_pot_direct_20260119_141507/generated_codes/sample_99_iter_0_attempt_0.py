import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select relevant columns
length_km = df['Length [km]'].astype(float)
basin_area_km2 = df['Drainage basin area [km2]'].astype(float)

# Create a bar chart with side-by-side bars
plt.figure(figsize=(14, 8))
bar_width = 0.35
index = range(len(length_km))

# Plot length and basin area
bars1 = plt.bar([i - bar_width/2 for i in index], length_km, bar_width, label='Length [km]', color='skyblue')
bars2 = plt.bar([i + bar_width/2 for i in index], basin_area_km2, bar_width, label='Drainage basin area [km²]', color='lightcoral')

# Labels and title
plt.xlabel('Rivers')
plt.ylabel('Values')
plt.title('Length of Rivers and Their Drainage Basin Areas')
plt.xticks([i for i in index], df['Name'], rotation=90)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)

plt.legend()
plt.tight_layout()
plt.show()