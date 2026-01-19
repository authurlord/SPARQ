import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Define the region columns
regions = ['mobiles', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales']

# Calculate total sales for each song
df['total'] = df[regions].sum(axis=1)

# Calculate percentages for each region per song
percentages = df[regions].div(df['total'], axis=0) * 100

# Plotting the percentage bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Set the positions and width for the bars
bar_width = 0.85
index = range(len(df))

# Create stacked bars for each region
bottom = [0] * len(df)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
for i, region in enumerate(regions):
    ax.bar(index, percentages[region], bar_width, label=region, bottom=bottom, color=colors[i])
    bottom = [b + p for b, p in zip(bottom, percentages[region])]

# Set labels and title
ax.set_xlabel('Song')
ax.set_ylabel('Percentage of Total Sales (%)')
ax.set_title('Percentage of Total Sales by Song and Region')
ax.set_xticks(index)
ax.set_xticklabels(df['song'], rotation=45)
ax.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()

# Final Answer: The chart is displayed.
Final Answer: Chart displayed