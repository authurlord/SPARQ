import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Exclude the 'total' column and keep only region columns
regions = df.columns[1:-1]  # 'mobiles' to 'wales'

# Calculate total sales across all regions
total_sales = df['total'].sum()

# Calculate percentage for each region per song
percentages = df[regions].div(df['total'], axis=0) * 100

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.1
index = range(len(df))

# Create bars for each region
for i, region in enumerate(regions):
    ax.bar([j + i * bar_width for j in index], percentages[region], bar_width, label=region)

ax.set_xlabel('Songs')
ax.set_ylabel('Percentage of Total Sales (%)')
ax.set_title('Percentage of Total Sales by Song and Region')
ax.set_xticks([j + bar_width * (len(regions) - 1) / 2 for j in index])
ax.set_xticklabels(df['song'], rotation=45)
ax.legend()
plt.tight_layout()
plt.show()

# Final Answer: The chart is displayed, no numerical answer needed.
print("Final Answer: Chart displayed")