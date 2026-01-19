import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row
record_high_row = df.iloc[0]  # First row corresponds to record high

# Extract Celsius values (inside parentheses)
celsius_values = []
for value in record_high_row[1:-1]:  # Exclude 'Month' and 'Year'
    celsius = value.split('(')[-1].strip(')')
    celsius_values.append(float(celsius))

# Months
months = df.columns[1:-1]  # Exclude 'Month' and 'Year'

# Plot bar chart
plt.figure(figsize=(12, 6))
plt.bar(months, celsius_values, color='skyblue')
plt.title('Record High Temperatures (°C) by Month')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final Answer: The bar chart is displayed
print("Final Answer: Bar chart displayed")