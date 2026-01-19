import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row
record_high_row = df.iloc[0]  # First row is 'Record high °F (°C)'

# Parse Celsius values from the string (extract value inside parentheses)
celsius_values = []
for cell in record_high_row[1:]:  # Skip the 'Month' column
    celsius = int(cell.split('(')[-1].split(')')[0])
    celsius_values.append(celsius)

# Months
months = df.columns[1:]  # Exclude 'Month' column

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(months, celsius_values, color='skyblue')
plt.title('Record High Temperatures (°C) by Month')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()