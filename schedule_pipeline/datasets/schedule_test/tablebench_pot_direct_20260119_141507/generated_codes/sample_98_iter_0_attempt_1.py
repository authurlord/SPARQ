import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and extract the relevant columns (removing the degree symbols and extracting just the numbers in inches)
def extract_inches(value):
    if isinstance(value, str):
        # Extract the part after "(inches)" and convert to float
        try:
            inches_part = value.split('(')[1].split(')')[0]
            return float(inches_part)
        except:
            return 0
    return 0

# Extract the months and relevant data
months = df.iloc[0, 1:]  # All months from Jan to Dec
months = months.reset_index(drop=True)

# Convert the data to inches
precip_inch = df.iloc[5, 1:].apply(extract_inches)
rainfall_inch = df.iloc[6, 1:].apply(extract_inches)
snowfall_inch = df.iloc[7, 1:].apply(extract_inches)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Define positions for the bars
x = range(len(months))
width = 0.2

# Create bars
ax.bar([p - width/2 for p in x], precip_inch, width, label='Precipitation', color='blue')
ax.bar([p + width/2 for p in x], rainfall_inch, width, label='Rainfall', color='green')
ax.bar([p + width for p in x], snowfall_inch, width, label='Snowfall', color='red')

# Labels and title
ax.set_xlabel('Month')
ax.set_ylabel('Inches')
ax.set_title('Trends in Precipitation, Rainfall, and Snowfall (in inches)')
ax.set_xticks(x)
ax.set_xticklabels(months, rotation=45)
ax.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()