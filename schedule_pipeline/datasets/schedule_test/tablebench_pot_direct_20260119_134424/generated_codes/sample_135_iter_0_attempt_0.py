import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Drop the 'total' column since we're computing percentages per song
regions = df.columns[1:-1]  # Exclude 'song' and 'total'

# Calculate percentage of each region's contribution per song
for region in regions:
    df[f'{region}_percent'] = (df[region] / df['total']) * 100

# Set up the bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Define positions for bars
x = range(len(df))
width = 0.8 / len(regions)  # Width for each bar

# Plot bars for each region
for i, region in enumerate(regions):
    ax.bar([pos + i * width for pos in x], df[f'{region}_percent'], width, label=region)

# Customize the chart
ax.set_xlabel('Song')
ax.set_ylabel('Percentage of Total Sales (%)')
ax.set_title('Percentage of Total Sales by Song and Region')
ax.set_xticks([pos + width * (len(regions) - 1) / 2 for pos in x])
ax.set_xticklabels(df['song'], rotation=45)
ax.legend()

plt.tight_layout()
plt.show()

# Final Answer: The chart is displayed.
print("Final Answer: Chart displayed")