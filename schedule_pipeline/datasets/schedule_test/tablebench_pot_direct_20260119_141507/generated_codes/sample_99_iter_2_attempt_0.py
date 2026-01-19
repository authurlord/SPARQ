import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert the required columns to numeric (handle any non-numeric entries)
df['Length [km]'] = pd.to_numeric(df['Length [km]'], errors='coerce')
df['Drainage basin area [km2]'] = pd.to_numeric(df['Drainage basin area [km2]'], errors='coerce')

# Drop rows with missing values in either column (to avoid plotting NaN)
df_clean = df.dropna(subset=['Length [km]', 'Drainage basin area [km2]'])

# Prepare data for plotting
rivers = df_clean['Name']
lengths = df_clean['Length [km]']
basin_areas = df_clean['Drainage basin area [km2]']

# Create a grouped bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Position of bars on x-axis
x = range(len(rivers))
width = 0.35

# Create bars
bars1 = ax.bar([i - width/2 for i in x], lengths, width, label='Length [km]', color='skyblue')
bars2 = ax.bar([i + width/2 for i in x], basin_areas, width, label='Drainage basin area [km2]', color='lightcoral')

# Add labels and title
ax.set_xlabel('River Name')
ax.set_ylabel('Value')
ax.set_title('Length and Drainage Basin Area of Rivers')
ax.set_xticks(x)
ax.set_xticklabels(rivers, rotation=45, ha='right')

# Add value labels on top of bars
def add_value_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

add_value_labels(bars1, lengths)
add_value_labels(bars2, basin_areas)

# Add legend
ax.legend()

# Improve layout
plt.tight_layout()

# Show plot
plt.show()