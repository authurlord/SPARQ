import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Parse the inch values from strings like '10.9\n(0.429)'
def extract_inches(value):
    match = re.search(r'\(([^)]+)\)', value)
    if match:
        return float(match.group(1))
    return 0.0

# Extract the months and the inch values for each category
months = df.iloc[0, 1:]  # Exclude 'Month' and 'Year'
precip_inch = [extract_inches(df.iloc[5, i]) for i in range(1, len(df.iloc[5, :]))]
rainfall_inch = [extract_inches(df.iloc[6, i]) for i in range(1, len(df.iloc[6, :]))]
snowfall_inch = [extract_inches(df.iloc[7, i]) for i in range(1, len(df.iloc[7, :]))]

# Create a bar chart
plt.figure(figsize=(14, 7))
x = range(len(months))
width = 0.2

plt.bar([i - width for i in x], precip_inch, width, label='Precipitation', color='blue')
plt.bar([i for i in x], rainfall_inch, width, label='Rainfall', color='green')
plt.bar([i + width for i in x], snowfall_inch, width, label='Snowfall', color='red')

plt.xlabel('Month')
plt.ylabel('Inches')
plt.title('Trends in Precipitation, Rainfall, and Snowfall (in inches)')
plt.xticks(x, months, rotation=45)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()