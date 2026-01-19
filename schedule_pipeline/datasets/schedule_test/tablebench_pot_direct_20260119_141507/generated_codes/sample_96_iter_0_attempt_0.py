import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row and parse Celsius values
record_high_celsius = df.iloc[0][1:]  # Skip first column 'Month'
# Parse the values like '76\n(24)' to extract 24
record_high_celsius_values = []
for val in record_high_celsius:
    # Extract the number in parentheses
    celsius_val = val.split(')')[1] if ')' in val else None
    if celsius_val:
        record_high_celsius_values.append(float(celsius_val.strip()))

# Months are in the first row of data (excluding 'Record high °F (°C)')
months = df.iloc[0][1:].index.tolist()  # This gives us the month names

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(months, record_high_celsius_values, color='skyblue')
plt.xlabel('Month')
plt.ylabel('Record High (°C)')
plt.title('Record High Temperatures in Celsius by Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the values for reference (optional)
print(f"Final Answer: {record_high_celsius_values}")