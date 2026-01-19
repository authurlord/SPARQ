import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Extract the inch values from each column (e.g., '10.9\n(0.429)' -> 0.429)
def extract_inches(value):
    match = re.search(r'\(([^)]+)\)', str(value))
    if match:
        return float(match.group(1))
    return 0.0

# Create a list of months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Extract inch values for each metric
precip_inch = df.iloc[5][months].apply(extract_inches).tolist()  # Precipitation mm (inches) row
rainfall_inch = df.iloc[6][months].apply(extract_inches).tolist()  # Rainfall mm (inches) row
snowfall_inch = df.iloc[7][months].apply(extract_inches).tolist()  # Snowfall cm (inches) row

# Create a bar chart
plt.figure(figsize=(14, 8))
x = range(len(months))

# Bar positions
width = 0.25

# Plot bars
plt.bar([p - width for p in x], precip_inch, width, label='Precipitation (inches)', color='blue', alpha=0.8)
plt.bar([p for p in x], rainfall_inch, width, label='Rainfall (inches)', color='green', alpha=0.8)
plt.bar([p + width for p in x], snowfall_inch, width, label='Snowfall (inches)', color='red', alpha=0.8)

# Labels and title
plt.xlabel('Month')
plt.ylabel('Inches')
plt.title('Trends in Precipitation, Rainfall, and Snowfall (in inches)')
plt.xticks(x, months, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# Final Answer: The bar chart has been plotted successfully.
Final Answer: chart_plotted