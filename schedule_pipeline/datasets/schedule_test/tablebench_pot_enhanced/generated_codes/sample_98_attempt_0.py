import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the month names and relevant columns
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
precip_inch = []
rainfall_inch = []
snowfall_inch = []

# Parse each row to extract inch values
for i in range(1, 7):  # rows 1 to 6 (excluding the header)
    row = df.iloc[i]
    # For each month, extract the inch value from the string
    for j, col in enumerate(months):
        val = row[col]
        if isinstance(val, str):
            # Extract the value in parentheses
            try:
                inch_val = float(val.split('(')[1].split(')')[0])
            except:
                inch_val = 0
        else:
            inch_val = 0
        if i == 1:  # Precipitation
            precip_inch.append(inch_val)
        elif i == 2:  # Rainfall
            rainfall_inch.append(inch_val)
        elif i == 6:  # Snowfall (row index 6 is snowfall)
            snowfall_inch.append(inch_val)

# Create a DataFrame for plotting
data = {
    'Month': months,
    'Precipitation (inches)': precip_inch,
    'Rainfall (inches)': rainfall_inch,
    'Snowfall (inches)': snowfall_inch
}
df_plot = pd.DataFrame(data)

# Plot the bar chart
plt.figure(figsize=(14, 8))
bar_width = 0.25
index = range(len(months))

# Create bars for each metric
plt.bar([i - bar_width for i in index], df_plot['Precipitation (inches)'], bar_width, label='Precipitation', color='blue')
plt.bar([i for i in index], df_plot['Rainfall (inches)'], bar_width, label='Rainfall', color='green')
plt.bar([i + bar_width for i in index], df_plot['Snowfall (inches)'], bar_width, label='Snowfall', color='red')

plt.xlabel('Month')
plt.ylabel('Inches')
plt.title('Trends in Precipitation, Rainfall, and Snowfall (in inches)')
plt.xticks(index, months, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()