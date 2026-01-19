import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert all numerical columns to numeric, handling any parsing issues
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN after conversion (in case of invalid entries)
df = df.dropna()

# Calculate percentage of total sales for each region per song
# We'll create a pivot table to get the percentage distribution
percentage_data = []
for index, row in df.iterrows():
    song = row['song']
    total = row['total']
    region_sales = row[['northern ireland', 'northern england', 'scotland', 'southern england', 'wales']].values
    percentages = [sale / total * 100 for sale in region_sales]
    percentage_data.append({
        'song': song,
        'northern ireland': percentages[0],
        'northern england': percentages[1],
        'scotland': percentages[2],
        'southern england': percentages[3],
        'wales': percentages[4]
    })

# Create a DataFrame from the percentage data
percentage_df = pd.DataFrame(percentage_data)

# Plot the percentage bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for each region
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# For each song, create a grouped bar chart
bar_width = 0.15
index = range(len(percentage_df))
x_pos = [i for i in index]

# Plot each region's percentage for each song
for i, region in enumerate(['northern ireland', 'northern england', 'scotland', 'southern england', 'wales']):
    values = percentage_df[region].values
    ax.bar([pos + i * bar_width for pos in x_pos], values, bar_width, label=region, color=colors[i])

# Customize plot
ax.set_xlabel('Song')
ax.set_ylabel('Percentage of Total Sales (%)')
ax.set_title('Percentage of Total Sales for Each Song by Region')
ax.set_xticks([i + bar_width * 2 for i in index])
ax.set_xticklabels(percentage_df['song'])
ax.legend()

plt.tight_layout()
plt.show()

# Final Answer is not a value but a chart, so we just confirm the chart was generated
Final Answer: chart_generated