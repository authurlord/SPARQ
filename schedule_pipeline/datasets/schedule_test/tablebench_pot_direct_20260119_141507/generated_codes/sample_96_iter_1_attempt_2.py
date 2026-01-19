import pandas as pd
import matplotlib.pyplot as plt

# Load the table
df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row
record_high_row = df.iloc[0]

# Parse the Celsius values from the parentheses
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
celsius_values = []

for cell in record_high_row[1:]:  # Skip the first column 'Month'
    # Extract the value in parentheses
    if isinstance(cell, str):
        # Find the part in parentheses
        celsius_str = cell.split('(')[1].split(')')[0]
        celsius_values.append(float(celsius_str))
    else:
        celsius_values.append(0)

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(months, celsius_values, color='skyblue')
plt.title('Record High Temperatures in Celsius Degrees')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the final answer as required (though no numerical answer, just visualization)
Final Answer: bar chart of record high in Celsius