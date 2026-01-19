import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Define the columns of interest
precip_col = 'Precipitation mm (inches)'
rain_col = 'Rainfall mm (inches)'
snow_col = 'Snowfall cm (inches)'

# Extract inches from the string values (e.g., "10.9 (0.429)" -> 0.429)
def extract_inches(value):
    match = re.search(r'\(([^)]+)\)', value)
    if match:
        return float(match.group(1))
    return 0.0

# Apply extraction to each column
df['precip_inches'] = df[precip_col].apply(extract_inches)
df['rain_inches'] = df[rain_col].apply(extract_inches)
df['snow_inches'] = df[snow_col].apply(extract_inches)

# The months are in the first row of the data, but the first row is labeled as "Record high °C (°F)", etc.
# So we extract the month names from the first row (excluding 'Year')
months = df.iloc[0, 1:13]  # Months from Jan to Dec

# Drop the first row (headers) since it contains labels
# We now have a DataFrame with months as index and the three metrics
data = df.iloc[1:, 1:13]  # Rows 1 to end, columns 1 to 12 (months)

# Reconstruct the data with correct month labels
month_names = [col for col in data.columns]
data_clean = data.applymap(extract_inches)

# Now, create a bar chart for each metric across months
fig, ax = plt.subplots(figsize=(14, 8))

# Bar positions
x = range(len(month_names))

# Width of bars
bar_width = 0.25

# Create bars
bars_precip = ax.bar([p - bar_width for p in x], data_clean.iloc[:, 0], bar_width, label='Precipitation', color='skyblue')
bars_rain = ax.bar([p for p in x], data_clean.iloc[:, 1], bar_width, label='Rainfall', color='lightgreen')
bars_snow = ax.bar([p + bar_width for p in x], data_clean.iloc[:, 2], bar_width, label='Snowfall', color='lightcoral')

# Labels and title
ax.set_xlabel('Month')
ax.set_ylabel('Inches')
ax.set_title('Trends in Precipitation, Rainfall, and Snowfall (in inches)')
ax.set_xticks(x)
ax.set_xticklabels(month_names, rotation=45)

# Add legend
ax.legend()

# Improve layout
plt.tight_layout()

# Show plot
plt.show()

# Final Answer: We have generated the bar chart as requested.
Final Answer: chart_generated