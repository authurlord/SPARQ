import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert the 'mobiles' column to numeric (though it seems irrelevant here)
# We focus on the regions and total for each song

# Drop the 'mobiles' column as it doesn't seem relevant to the analysis
df = df.drop(columns=['mobiles'])

# Convert all region columns to numeric
region_cols = ['northern ireland', 'northern england', 'scotland', 'southern england', 'wales']
df[region_cols] = df[region_cols].apply(pd.to_numeric, errors='coerce')

# Calculate percentage of total sales for each region per song
df['total'] = pd.to_numeric(df['total'], errors='coerce')
df['total'] = df['total'].fillna(0)

# Create a new dataframe for percentages
percentages = df.copy()
for col in region_cols:
    percentages[col] = (percentages[col] / percentages['total']) * 100

# Now we want to plot a stacked bar chart: one bar per song, with segments for each region
# We will use the percentages to create the chart

# Set up the plot
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = df.index

# Create the stacked bar chart
bars = []
for i, col in enumerate(region_cols):
    bars.append(plt.bar(index, percentages[col], label=col, width=bar_width))

# Add labels and title
plt.xlabel('Song')
plt.ylabel('Percentage of Total Sales')
plt.title('Percentage of Total Sales for Each Song by Region')
plt.xticks(index, df['song'], rotation=45)

# Add legend
plt.legend(title='Region')

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# Since the question asks to draw the chart, we don't print a final answer in the format unless required.
# But if required, we could print a message.