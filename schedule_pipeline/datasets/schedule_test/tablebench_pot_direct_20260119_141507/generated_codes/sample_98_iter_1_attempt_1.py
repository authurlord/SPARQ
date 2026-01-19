import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Extract the month names from the first row (excluding 'Year')
months = df.iloc[0, 1:13].index.tolist()
# Remove 'Year' and keep only the month names
months = [month for month in months if month != 'Year']

# Define the columns of interest
precip_col = 'Precipitation mm (inches)'
rain_col = 'Rainfall mm (inches)'
snow_col = 'Snowfall cm (inches)'

# Function to extract inch value from string like "10.9\n(0.429)"
def extract_inches(value):
    match = re.search(r'\(([^)]+)\)', value)
    if match:
        return float(match.group(1))
    return 0.0

# Extract inch values for each month
precip_inch = df[precip_col][1:].apply(extract_inches)
rain_inch = df[rain_col][1:].apply(extract_inches)
snow_inch = df[snow_col][1:].apply(extract_inches)

# Ensure all series have the same length as months
# We have 12 months, so align with months list
months = [str(m).strip() for m in months]
precip_inch = precip_inch.reset_index(drop=True)
rain_inch = rain_inch.reset_index(drop=True)
snow_inch = snow_inch.reset_index(drop=True)

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Month': months,
    'Precipitation (inches)': precip_inch,
    'Rainfall (inches)': rain_inch,
    'Snowfall (inches)': snow_inch
})

# Plot bar chart
plt.figure(figsize=(14, 8))
bar_width = 0.25
index = range(len(months))

# Create bars
plt.bar([i - bar_width for i in index], plot_data['Precipitation (inches)'], bar_width, label='Precipitation', color='skyblue')
plt.bar([i for i in index], plot_data['Rainfall (inches)'], bar_width, label='Rainfall', color='lightgreen')
plt.bar([i + bar_width for i in index], plot_data['Snowfall (inches)'], bar_width, label='Snowfall', color='pink')

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