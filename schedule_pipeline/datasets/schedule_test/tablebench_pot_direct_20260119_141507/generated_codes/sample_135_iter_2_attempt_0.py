import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert all relevant columns to numeric, handling any parsing errors
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN after conversion (in case of invalid entries)
df = df.dropna()

# Calculate percentage of total sales for each region per song
# We'll create a pivot table to show percentages
pivot = df.set_index('song')[['mobiles', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales']].T
pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100  # Normalize to percentages

# Plot the percentage bar chart
pivot.plot(kind='bar', figsize=(12, 8))
plt.title('Percentage of Total Sales for Each Song by Region')
plt.ylabel('Percentage (%)')
plt.xlabel('Song')
plt.xticks(rotation=45)
plt.legend(title='Region')
plt.tight_layout()
plt.show()

# Since the question asks to draw the chart, we don't need to print a final answer value.
# But if required, we could summarize the max percentage or something, but it's not asked.