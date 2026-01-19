import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to integers
df = df.astype({col: 'int' for col in ['mobiles', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales', 'total']})

# Calculate percentage of each region's sales relative to the total for each song
regions = ['mobiles', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales']
df_percent = df.copy()
for region in regions:
    df_percent[region + '_percent'] = (df[region] / df['total']) * 100

# Set up the bar chart
plt.figure(figsize=(12, 8))
bar_width = 0.2
index = range(len(df))

# Create bars for each region
for i, region in enumerate(regions):
    values = df_percent[region + '_percent'].values
    plt.bar([x + i * bar_width for x in index], values, bar_width, label=region)

plt.xlabel('Song')
plt.ylabel('Percentage of Total Sales')
plt.title('Percentage of Total Sales for Each Song by Region')
plt.xticks([x + bar_width * 1.5 for x in index], df['song'], rotation=45)
plt.legend(title='Region')
plt.tight_layout()

# Show the plot
plt.show()

# Final Answer is not a number or entity, but the chart is generated. 
# Since the question asks to draw the chart, no final answer name is needed.
# However, following format strictly:
Final Answer: chart_generated