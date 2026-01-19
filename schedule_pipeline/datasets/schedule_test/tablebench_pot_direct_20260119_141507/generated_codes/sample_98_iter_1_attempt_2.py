import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Extract the months (Jan to Dec)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Function to extract the inch value from string like "X.X (Y.Y)"
def extract_inches(value):
    if pd.isna(value):
        return 0
    match = re.search(r'\(([^)]+)\)', str(value))
    if match:
        inch_str = match.group(1)
        return float(inch_str)
    return 0

# Extract and clean the inch values
precip_inch = df['Precipitation mm (inches)'].apply(extract_inches)
rainfall_inch = df['Rainfall mm (inches)'].apply(extract_inches)
snowfall_inch = df['Snowfall cm (inches)'].apply(extract_inches)

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Month': months,
    'Precipitation (in)': precip_inch,
    'Rainfall (in)': rainfall_inch,
    'Snowfall (in)': snowfall_inch
})

# Plot bar chart
plt.figure(figsize=(14, 8))
bar_width = 0.2
index = range(len(months))

# Bar plots for each category
plt.bar([i - bar_width for i in index], plot_data['Precipitation (in)'], width=bar_width, label='Precipitation', edgecolor='black')
plt.bar([i for i in index], plot_data['Rainfall (in)'], width=bar_width, label='Rainfall', edgecolor='black')
plt.bar([i + bar_width for i in index], plot_data['Snowfall (in)'], width=bar_width, label='Snowfall', edgecolor='black')

# Labels and title
plt.xlabel('Month')
plt.ylabel('Inches')
plt.title('Trends in Precipitation, Rainfall, and Snowfall (in inches)')
plt.xticks(index, months, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()