import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert all numeric columns to integers
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate percentage of total sales for each song in each region
# We'll create a pivot table with songs as index and regions as columns
pivot = df.set_index('song').transpose()

# Compute percentages for each song (row-wise)
percentage_data = []
for song in df['song']:
    row = df[df['song'] == song]
    total_sales = row['total'].values[0]
    region_sales = row.iloc[:, 1:-1].values[0]  # exclude 'total' and 'mobiles'
    percentages = [round((sale / total_sales) * 100, 1) for sale in region_sales]
    percentage_data.append(percentages)

# Now, create a bar chart for each song
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for plotting
songs = df['song'].tolist()
region_names = df.columns[1:-1].tolist()  # excluding 'total' and 'mobiles'

# Plot each song's percentage distribution
for i, song in enumerate(songs):
    bars = ax.bar(region_names, percentage_data[i], label=song, alpha=0.8)

# Add labels and title
ax.set_xlabel('Region')
ax.set_ylabel('Percentage of Total Sales (%)')
ax.set_title('Percentage of Total Sales for Each Song by Region')
ax.legend(title='Song')

# Improve layout
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()

print("Final Answer: Chart generated")