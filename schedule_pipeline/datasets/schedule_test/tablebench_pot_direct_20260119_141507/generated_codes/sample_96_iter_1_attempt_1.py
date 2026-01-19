import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row and parse the Celsius values
record_high_celsius = df.iloc[0][1:]  # Take all values from the second column onwards
# Clean and extract the numbers in parentheses
celsius_values = []
for cell in record_high_celsius:
    # Find the number in parentheses
    celsius_str = cell.split('(')[1].strip().split(')')[0]
    celsius_values.append(float(celsius_str))

# Create a bar chart
months = df.iloc[0][:-1]  # Months excluding 'Year'
plt.figure(figsize=(10, 6))
plt.bar(months, celsius_values, color='skyblue')
plt.title('Record High Temperatures in Celsius Degrees')
plt.xlabel('Month')
plt.ylabel('Record High (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the values for verification (optional)
print(f"Final Answer: {celsius_values}")