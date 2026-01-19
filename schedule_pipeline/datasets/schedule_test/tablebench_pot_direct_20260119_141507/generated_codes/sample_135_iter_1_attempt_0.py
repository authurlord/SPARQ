import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to integers
df[['mobiles', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales']] = df[['mobiles', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales']].apply(pd.to_numeric, errors='coerce')

# Remove any NaNs due to conversion
df = df.dropna()

# Create a list of regions
regions = ['mobiles', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales']

# For each song, compute the percentage of sales in each region relative to the total
# We'll create a pivot table or use a loop to compute percentages
percentage_data = []

for idx, row in df.iterrows():
    song = row['song']
    total = row['total']
    region_sales = row[regions]
    # Compute percentages
    percentages = (region_sales / total * 100).round(1)
    percentage_data.append({
        'song': song,
        'mobiles': percentages[0],
        'northern ireland': percentages[1],
        'northern england': percentages[2],
        'scotland': percentages[3],
        'southern england': percentages[4],
        'wales': percentages[5]
    })

# Convert to DataFrame
percentage_df = pd.DataFrame(percentage_data)

# Set up the bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define positions for the bars
bar_width = 0.15
index = range(len(percentage_df))
x_pos = [i for i in index]

# Plot each region's percentage
for i, region in enumerate(regions):
    values = percentage_df[region]
    ax.bar([pos + i * bar_width for pos in x_pos], values, bar_width, label=region)

# Customize labels and title
ax.set_xlabel('Song')
ax.set_ylabel('Percentage of Total Sales (%)')
ax.set_title('Percentage of Total Sales for Each Song by Region')
ax.set_xticks([pos + bar_width * 1.5 for pos in index])
ax.set_xticklabels(percentage_df['song'], rotation=45)

# Add legend
ax.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()