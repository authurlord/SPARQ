import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Select only the relevant columns
df_filtered = df[['Length [km]', 'Drainage basin area [km2]']]

# Drop rows with NaN values in either column to avoid plotting invalid data
df_filtered = df_filtered.dropna()

# Create a grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Position of bars on the x-axis
x = range(len(df_filtered))
width = 0.35

# Length and area bars
bars1 = ax.bar([i - width/2 for i in x], df_filtered['Length [km]'], width, label='Length [km]')
bars2 = ax.bar([i + width/2 for i in x], df_filtered['Drainage basin area [km2]'], width, label='Drainage basin area [km2]')

# Labels and title
ax.set_xlabel('Rivers')
ax.set_ylabel('Values')
ax.set_title('Length of Rivers and Their Drainage Basin Areas')
ax.set_xticks(x)
ax.set_xticklabels([row[0] for row in df_filtered.values], rotation=45)

# Add value labels on top of bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)

# Legend
ax.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()