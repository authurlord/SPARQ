import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Extract the month names and the values for each metric
months = df.iloc[0, 1:]  # Exclude 'Year' and 'Record high °C (°F)' etc., take from column 1 onwards
months = months.dropna().index.tolist()  # Get month names

# Function to extract inches from string like "10.9\n(0.429)"
def extract_inches(value):
    match = re.search(r'\(([^)]+)\)', value)
    if match:
        return float(match.group(1))
    return 0.0

# Extract the relevant rows (precipitation, rainfall, snowfall)
precip_inch = df.iloc[5, 1:].apply(extract_inches)
rainfall_inch = df.iloc[6, 1:].apply(extract_inches)
snowfall_inch = df.iloc[7, 1:].apply(extract_inches)

# Create a figure and axis
plt.figure(figsize=(14, 8))

# Create grouped bar chart
x = range(len(months))
width = 0.25

plt.bar([i - width for i in x], precip_inch, width, label='Precipitation', color='blue', alpha=0.8)
plt.bar([i for i in x], rainfall_inch, width, label='Rainfall', color='green', alpha=0.8)
plt.bar([i + width for i in x], snowfall_inch, width, label='Snowfall', color='red', alpha=0.8)

# Labels and title
plt.xlabel('Month')
plt.ylabel('Inches')
plt.title('Trends in Precipitation, Rainfall, and Snowfall (in inches)')
plt.xticks([i for i in x], months, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()